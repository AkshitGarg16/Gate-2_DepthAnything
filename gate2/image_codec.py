"""ROS image messages <-> BGR arrays, without cv_bridge.

cv_bridge is deliberately not used: ROS Humble's cv_bridge is built against
NumPy 1.x and fails under NumPy 2 with "_ARRAY_API not found" -- inside the
subscription callback, where the error never reaches the terminal and the node
just looks hung. A sensor_msgs/Image is a flat buffer plus a row stride, so
this is a reshape.

Three transports are handled:
  sensor_msgs/Image            raw pixels
  sensor_msgs/CompressedImage  JPEG/PNG, decoded with OpenCV
  ffmpeg_image_transport_msgs/FFMPEGPacket
                               H.264/HEVC packets (what the FCB camera records),
                               decoded in-process with PyAV -- so neither an
                               image_transport republisher nor a GPU decoder
                               is needed.
"""
from __future__ import annotations

import cv2
import numpy as np

from .utils import log

_ENC = {"bgr8": (3, np.uint8), "rgb8": (3, np.uint8), "mono8": (1, np.uint8),
        "bgra8": (4, np.uint8), "rgba8": (4, np.uint8), "8UC1": (1, np.uint8),
        "8UC3": (3, np.uint8), "mono16": (1, np.uint16), "16UC1": (1, np.uint16)}


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def imgmsg_to_bgr(m):
    if m.encoding not in _ENC:
        raise ValueError(f"unsupported image encoding {m.encoding!r}")
    ch, dt = _ENC[m.encoding]
    stride = m.step // np.dtype(dt).itemsize
    a = np.frombuffer(m.data, dt).reshape(m.height, stride)[:, :m.width * ch]
    a = a.reshape(m.height, m.width, ch)
    if dt == np.uint16:
        a = cv2.convertScaleAbs(a, alpha=255.0 / 65535.0)
    if ch == 1:
        return cv2.cvtColor(a.reshape(m.height, m.width), cv2.COLOR_GRAY2BGR)
    conv = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR, "bgra8": cv2.COLOR_BGRA2BGR}
    if m.encoding in conv:
        return cv2.cvtColor(a, conv[m.encoding])
    return a.copy()   # frombuffer is read-only


def compressed_to_bgr(m):
    img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode CompressedImage ({m.format})")
    return img


def bgr_to_imgmsg(bgr, stamp, frame_id="camera"):
    from sensor_msgs.msg import Image
    m = Image()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.height, m.width = bgr.shape[:2]
    m.encoding = "bgr8"
    m.is_bigendian = 0
    m.step = m.width * 3
    m.data = np.ascontiguousarray(bgr).tobytes()
    return m


class FFmpegPacketDecoder:
    """Decodes a stream of FFMPEGPacket messages into (bgr, header_stamp) frames.

    Every packet must be fed in order -- an inter-coded stream cannot skip --
    so the caller decodes all of them and drops *decoded frames* instead.
    Packets before the first keyframe produce nothing (the first ~20 on the
    fcb bags). Frames are matched back to their packet's header stamp by pts.
    """

    def __init__(self):
        self.ctx = None
        self.codec = None
        self.pts_stamp = {}
        self.errors = 0
        self.seen_key = False      # packets before the first keyframe cannot decode
        self.skipped = 0

    def _open(self, encoding):
        import av
        enc = (encoding or "").split(";")[0].lower()
        name = "hevc" if ("hevc" in enc or "265" in enc) else \
               "h264" if "264" in enc else "av1" if "av1" in enc else enc
        self.ctx = av.CodecContext.create(name, "r")
        self.codec = name
        log("info", "source", f"decoding FFMPEGPacket stream with PyAV codec '{name}'")

    def decode(self, msg):
        import av
        if self.ctx is None:
            self._open(msg.encoding)
        if not self.seen_key:
            # joining a stream mid-way (live, or a bag after a seek): wait for a
            # keyframe. If the encoder never flags one, give up waiting after
            # 300 packets and let the decoder find its own way in.
            if msg.flags & 1 or self.skipped >= 300:
                self.seen_key = True
            else:
                self.skipped += 1
                return []
        t = stamp_s(msg.header)
        self.pts_stamp[msg.pts] = t
        if len(self.pts_stamp) > 512:
            for k in sorted(self.pts_stamp)[:256]:
                del self.pts_stamp[k]
        pkt = av.Packet(bytes(msg.data))
        pkt.pts = msg.pts
        out = []
        try:
            for f in self.ctx.decode(pkt):
                out.append((f.to_ndarray(format="bgr24"), self.pts_stamp.get(f.pts, t)))
        except av.error.FFmpegError:
            self.errors += 1   # normal until the first keyframe when joining mid-stream
        return out


def make_image_decoder(type_name):
    """Return f(msg) -> list[(bgr, stamp)] for a ROS image message type string."""
    if type_name.endswith("FFMPEGPacket"):
        dec = FFmpegPacketDecoder()
        return dec.decode, "ffmpeg"
    if type_name.endswith("CompressedImage"):
        return (lambda m: [(compressed_to_bgr(m), stamp_s(m.header))]), "compressed"
    if type_name.endswith("/Image"):
        return (lambda m: [(imgmsg_to_bgr(m), stamp_s(m.header))]), "raw"
    raise ValueError(f"not an image type: {type_name}")
