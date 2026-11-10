# Vendored copy of GATE_1_PIPELINE/deployment_gate_1/coordinate_transformer.py (2026-10-10).
# Used only when the Gate-1 checkout cannot be found; keep it identical to the original.
"""
Coordinate transformation utilities for converting pixel coordinates to GPS coordinates.
"""

import numpy as np
import math


class CoordinateTransformer:
    
    
    def __init__(self, intrinsic_matrix):
        
        self.K = intrinsic_matrix
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]
        
        self.EARTH_RADIUS = 6378137.0
        
    def pixel_to_camera_ray(self, u, v, zoom=1.0):
        # Optical zoom scales the focal length; the principal point is unchanged.
        # `zoom` is the ratio relative to the zoom the intrinsics were calibrated at.
        if zoom is None or zoom <= 0:
            zoom = 1.0
        X_cam = (u - self.cx) / (self.fx * zoom)
        Y_cam = (v - self.cy) / (self.fy * zoom)
        Z_cam = 1.0
        

        ray = np.array([X_cam, Y_cam, Z_cam])
        ray = ray / np.linalg.norm(ray)
        
        return ray
    
    def camera_to_body_frame(self, point_camera):
      
        R_body_cam = np.array([
            [0,  -1,  0],   
            [1,   0,  0],   
            [0,   0,  1]    
        ])
        
        point_body = R_body_cam @ point_camera
        return point_body
    
    


    def body_to_world_frame(self, point_body, roll, pitch, yaw):
        pitch = -pitch 
        R_roll = np.array([
            [1,           0,            0],
            [0,  np.cos(roll), -np.sin(roll)],
            [0,  np.sin(roll),  np.cos(roll)]
        ])
        R_pitch = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0,              1,             0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])

        # Correct order for NED: Yaw applied last (outermost)
        R_world_body = R_yaw @ R_pitch @ R_roll
        return R_world_body @ point_body
    
    def find_ground_intersection(self, ray_world, drone_altitude, ground_altitude=0.0):
     
        altitude_diff = drone_altitude - ground_altitude

        if ray_world[2] <= 0:

            return None
        
  
        scale = altitude_diff / ray_world[2]
  
        intersection = ray_world * scale
        
        return intersection
    
    def ned_to_gps(self, north_offset, east_offset, drone_lat, drone_lon):
      
        lat_rad = math.radians(drone_lat)
    
        meters_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + \
                            1.175 * math.cos(4 * lat_rad) - 0.0023 * math.cos(6 * lat_rad)
        meters_per_deg_lon = 111412.84 * math.cos(lat_rad) - \
                            93.5 * math.cos(3 * lat_rad) + 0.118 * math.cos(5 * lat_rad)
        
       
        delta_lat = north_offset / meters_per_deg_lat
        delta_lon = east_offset / meters_per_deg_lon

        target_lat = drone_lat + delta_lat
        target_lon = drone_lon + delta_lon
        
        return target_lat, target_lon
    
    def pixel_to_gps(self, u, v, drone_lat, drone_lon, drone_altitude,
                 roll, pitch, yaw, ground_altitude=0.0, zoom=1.0):
        
    # ---------------------------------------------------
    # 1. Camera offset in BODY frame (NED body)
        # X=Forward (North), Y=Right (East), Z=Down
        # ---------------------------------------------------
        camera_offset_body = np.array([0.29, 0.03, 0.06])


    
        # ---------------------------------------------------
        # 2. Ray in camera → body → world (NED)
        # ---------------------------------------------------
        ray_camera = self.pixel_to_camera_ray(u, v, zoom)
        ray_body = self.camera_to_body_frame(ray_camera)
        ray_world = self.body_to_world_frame(ray_body, roll, pitch, yaw)
    
        # ---------------------------------------------------
        # 3. Rotate camera offset into world (NED)
        # ---------------------------------------------------
        offset_world = self.body_to_world_frame(
            camera_offset_body, roll, pitch, yaw
        )
    
        # ---------------------------------------------------
        # 4. Convert altitude (positive UP) → NED (positive DOWN)
        # ---------------------------------------------------
        drone_z_ned = -(drone_altitude - ground_altitude)
    
        # Camera world Z in NED
        camera_z_ned = drone_z_ned + offset_world[2]
    
        # If ray not pointing downward, skip
       # print("ray_world:", ray_world)
       
        if ray_world[2] <= 0:
            return None
    
        # ---------------------------------------------------
        # 5. Solve intersection with ground plane (z = 0 in NED)
        # ---------------------------------------------------
        scale = -camera_z_ned / ray_world[2]
    
        intersection_ned = ray_world * scale
    
        # ---------------------------------------------------
        # 6. Add horizontal camera offset
        # ---------------------------------------------------
        north_offset = offset_world[0] + intersection_ned[0]
        east_offset  = offset_world[1] + intersection_ned[1]
    
        # ---------------------------------------------------
        # 7. Convert NED offsets to GPS
        # ---------------------------------------------------
        target_lat, target_lon = self.ned_to_gps(
            north_offset, east_offset,
            drone_lat, drone_lon
        )
    
        return target_lat, target_lon
