# services/location_service.py
"""
Dynamic geofencing - automatically detects server location
Students must be within range of wherever the system is running
"""

import math
import requests


class LocationService:
    """Dynamic geofencing based on server's current location"""
    
    def __init__(self):
        self.server_lat = None
        self.server_lon = None
        self.MAX_DISTANCE_METERS = 100  # 100 meter radius
        self.location_initialized = False
    
    def get_server_location_from_ip(self):
        """
        Get server's approximate location using IP geolocation
        This works but has ~50-500m accuracy
        """
        try:
            response = requests.get('https://ipapi.co/json/', timeout=5)
            data = response.json()
            
            self.server_lat = data.get('latitude')
            self.server_lon = data.get('longitude')
            self.location_initialized = True
            
            print(f"📍 Server location detected: ({self.server_lat}, {self.server_lon})")
            print(f"   City: {data.get('city')}, Region: {data.get('region')}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to get server location: {e}")
            return False
    
    def set_server_location_manual(self, lat, lon):
        """
        Manually set server location (more accurate)
        Use this if you know the exact classroom coordinates
        """
        self.server_lat = lat
        self.server_lon = lon
        self.location_initialized = True
        print(f"📍 Server location set manually: ({lat}, {lon})")
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance using Haversine formula (in meters)"""
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def verify_location(self, user_lat, user_lon):
        """
        Verify if student is within range of server
        Returns: (is_valid, distance, message)
        """
        # Initialize server location if not done
        if not self.location_initialized:
            if not self.get_server_location_from_ip():
                return False, None, "❌ Could not detect server location"
        
        if user_lat is None or user_lon is None:
            return False, None, "❌ Student location not provided"
        
        if self.server_lat is None or self.server_lon is None:
            return False, None, "❌ Server location not available"
        
        # Calculate distance
        distance = self.calculate_distance(
            self.server_lat,
            self.server_lon,
            user_lat,
            user_lon
        )
        
        # Check if within range
        if distance <= self.MAX_DISTANCE_METERS:
            return True, distance, f"✅ Location verified ({distance:.1f}m from system)"
        else:
            return False, distance, f"❌ Too far from system ({distance:.1f}m away, max {self.MAX_DISTANCE_METERS}m)"
    
    def get_server_info(self):
        """Get server location info"""
        if not self.location_initialized:
            self.get_server_location_from_ip()
        
        return {
            'latitude': self.server_lat,
            'longitude': self.server_lon,
            'radius_meters': self.MAX_DISTANCE_METERS,
            'initialized': self.location_initialized
        }
    
    def update_radius(self, radius_meters):
        """Update acceptable distance radius"""
        self.MAX_DISTANCE_METERS = radius_meters
        print(f"📍 Radius updated to {radius_meters}m")
