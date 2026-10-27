import numpy as np
from bluesky.tools.geo import kwikqdrdist
from bluesky.tools.aero import nm
from bluesky.tools.misc import degto180
from bluesky.tools import areafilter
from typing import List

def bound_angle_positive_negative_180(angle_deg: float) -> float:
    """ maps any angle in degrees to the [-180,180] interval 
    Parameters
    __________
    angle_deg: float
        angle that needs to be mapped (in degrees)
    
    Returns
    __________
    angle_deg: float
        input angle mapped to the interval [-180,180] (in degrees)
    """

    if angle_deg > 180:
        return -(360 - angle_deg)
    elif angle_deg < -180:
        return (360 + angle_deg)
    else:
        return angle_deg

def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: latitude of the reference point, in degrees
    lon: longitude of the referemce point, in degrees
    d: target distance from the reference point, in km
    bearing: (true) heading, in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from the reference point, in degrees
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    a = np.radians(bearing)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d/R) + np.cos(lat1) * np.sin(d/R) * np.cos(a))
    lon2 = lon1 + np.arctan2(
        np.sin(a) * np.sin(d/R) * np.cos(lat1),
        np.cos(d/R) - np.sin(lat1) * np.sin(lat2)
    )
    return np.degrees(lat2), np.degrees(lon2)

def random_point_on_circle(radius: float, rng=None) -> np.array:
    """ Get a random point on a circle circumference with given radius
    Parameters
    __________
    radius: float
        radius for the circle
    rng: np.random.Generator, optional
        generator to sample from (pass the environment's self.np_random for reproducible resets).
        Defaults to the global numpy generator.
    
    Returns
    __________
    point: np.array
        randomly sampled point
    """
    rng = np.random if rng is None else rng
    alpha = 2 * np.pi * rng.uniform(0., 1.)
    x = radius * np.cos(alpha)
    y = radius * np.sin(alpha)
    return np.array([x, y])

def sort_points_clockwise(vertices: np.array) -> np.array:
    """ Sort the points in clockwise order
    Parameters
    __________
    vertices: np.array
        array of points
    
    Returns
    __________
    sorted_vertices: np.array
        sorted array of points
    """
    sorted_vertices = [vertices[i] for i in np.argsort([np.arctan2(v[1], v[0]) for v in vertices])]

    return sorted_vertices   

def polygon_area(vertices: np.array) -> float:
    """ Calculate the area of a polygon given the vertices
    Parameters
    __________
    vertices: np.array
        array of vertices of the polygon
    
    Returns
    __________
    area: float
        area of the polygon
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Wrap around to the first vertex
        area += x1 * y2 - y1 * x2
    area = np.abs(area) / 2.0
    return area

def nm_to_latlong(center: np.array, point: np.array) -> np.array:
    """ Convert a point in nm to lat/long coordinates
    Parameters
    __________
    center: np.array
        center point of the conversion
    point: np.array
        point to be converted
    
    Returns
    __________
    latlong: np.array
        converted point in lat/long coordinates
    """
    lat = center[0] + (point[0] / 60)
    lon = center[1] + (point[1] / (60 * np.cos(np.radians(center[0]))))
    return np.array([lat, lon])

def latlong_to_nm(center: np.array, point: np.array) -> np.array:
    """ Convert a point in lat/long coordinates to nm
    Parameters
    __________
    center: np.array
        center point of the conversion
    point: np.array
        point to be converted
    
    Returns
    __________
    nm: np.array
        converted point in nm
    """
    x = (point[0] - center[0]) * 60
    y = (point[1] - center[1]) * 60 * np.cos(np.radians(center[0]))
    return np.array([x, y])

def euclidean_distance(point1: np.array, point2: np.array) -> float:
    """ Calculate the euclidean distance between two points
    Parameters
    __________
    point1: np.array
        [x, y] of the first point
    point2: np.array
        [x, y] of the second point
        
    Returns
    __________
    distance: float
        euclidean distance between the two points
    """
    return np.sqrt(np.sum((point2 - point1)**2))

def get_hdg(point1: np.array, point2: np.array) -> float:
    """ Calculate the heading from point1 to point2
    Parameters
    __________
    point1: np.array
        [lat, lon] of the first point 
    point2: np.array
        [lat, lon] of the second point
    
    Returns
    __________
    hdg: float
        heading from point1 to point2
    """
    
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)
    
    delta_lon = lon2 - lon1
    
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    
    hdg = np.degrees(np.arctan2(x, y))
    
    hdg = (hdg + 360) % 360 # Convert back to [0, 360] interval
    
    return hdg


def distance_to_obstacles(ac_lat, ac_lon, centre_lat, centre_lon, radius_nm):
    """ Distance from one or more points to the nearest obstacle, where every obstacle is approximated by the circle
    it was generated in (centre and radius R). The polygon vertices lie on that circle, so the circle contains the polygon
    and this distance is never larger than the true distance to the polygon.
    This is cheaper than the exact distance to the polygon boundary: one distance per point-obstacle pair instead of one per edge.

    Input:
        ac_lat, ac_lon: scalar or array shape (P,) -> aircraft latitude(s) and longitude(s) [degrees, degrees]
        centre_lat, centre_lon: scalar or array shape (K,) -> obstacle centre(s) latitude(s) and longitude(s) [degrees, degrees]
        radius_nm: scalar or array shape (K,) -> obstacle(s) radius [NM]
    Returns:
        distance_km: np.ndarray, shape (P,) -> distance to the circle edge of the nearest obstacle [km]; negative if inside a circle. """
    
    lat = np.atleast_1d(np.asarray(ac_lat, dtype=float))
    lon = np.atleast_1d(np.asarray(ac_lon, dtype=float))
    c_lat = np.atleast_1d(np.asarray(centre_lat, dtype=float))
    c_lon = np.atleast_1d(np.asarray(centre_lon, dtype=float))
    radius = np.atleast_1d(np.asarray(radius_nm, dtype=float))
    lat_a, lon_a, lat_b, lon_b = np.broadcast_arrays(lat[:, None], lon[:, None], c_lat[None, :], c_lon[None, :])
    _, dist_nm = kwikqdrdist(lat_a.ravel(), lon_a.ravel(), lat_b.ravel(), lon_b.ravel())  # NM, shape (P*K,)
    dist_nm = dist_nm.reshape(lat_a.shape)
    return (dist_nm - radius[None, :]).min(axis=1) * nm / 1000.  # NM -> km


def polygon_observation(ac_lat: float, ac_lon: float, ac_hdg: float, ac_alt: float, area_name: str, vertices):
    """ Observation of a (convex) polygonal obstacle as seen from the aircraft.
    The observation consists of:
     - the distance to the nearest point of the polygon (positive when the aircraft is outside the polygon, negative when inside), 
     - the bearing of that point relative to the aircraft heading,
     - the bearings of the left-most and right-most tangents of the polygon relative to the aircraft heading.

    Parameters
    __________
    ac_lat, ac_lon: float
        position of the aircraft (degrees)
    ac_hdg: float
        heading of the aircraft (degrees)
    ac_alt: float
        altitude of the aircraft (m)
    area_name: str
        name of the polygon in the BlueSky area filter
    vertices: array-like, shape (N, 2)
        vertices of the polygon as [lat, lon] (degrees)

    Returns
    __________
    clearance_km: float
        distance (positive when the aircraft is outside the polygon, negative when inside) from the aircraft to the nearest point of the polygon boundary (km)
    cos_bearing, sin_bearing: float
        cosine and sine of the bearing (heading − qdr) of the nearest point relative to the aircraft heading. (-)
    left_edge, right_edge: float
        angle between the heading and the left-most and right-most tangent of the polygon, (heading - qdr)/180 in [-1, 1]. (-)
        If the aircraft is inside the polygon the polygon spans all directions: left_edge = 1, right_edge = -1.
    """
    v = np.asarray(vertices, dtype=float).reshape(-1, 2)

    # bearing (0 is true north, values 0 to 360) [deg] and distance [NM] from the aircraft to each vertex
    qdr_vertices, dist_vertices = kwikqdrdist(ac_lat, ac_lon, v[:, 0], v[:, 1])
    # convert to local coordinates in NM
    x = dist_vertices * np.sin(np.radians(qdr_vertices)) # [NM]
    y = dist_vertices * np.cos(np.radians(qdr_vertices)) # [NM]
    # shift the vertices of one position to later make the segments
    xn, yn = np.roll(x, -1), np.roll(y, -1)

    # vectors between one vertex of the polygon and the next one
    ex, ey = xn - x, yn - y
    # squared length of each edge. 1e-12 is used to avoid division by zero in case of duplicate vertices (which should not happen in theory)
    seg_len2 = np.maximum(ex * ex + ey * ey, 1e-12)
    # projection of the aircraft position onto a line defined by each edge. The projection is clipped to the actual boundary of the edge, so t=0 means the projection is either vertex i or further, t=1 means the projection is either vertex i+1 or further.
    t = np.clip(-(x * ex + y * ey) / seg_len2, 0.0, 1.0)
    # coordinates of the nearest point (corresponding to the projection t) on each edge between one vertex of the polygon and the next one
    px, py = x + t * ex, y + t * ey
    # distance from the aircraft to each nearest point on the segments (either a vertex or on one of the edges)
    dist = np.hypot(px, py)
    # index of the edge on the polygon boundary containing the nearest point
    k = int(np.argmin(dist))
    # bearing to the nearest point from the aircraft (0 is true north, values 0 to 360) [deg]
    qdr_nearest = np.degrees(np.arctan2(px[k], py[k])) % 360.0

    # check if the aircraft is inside the polygon
    inside = bool(areafilter.checkInside(area_name, np.array([ac_lat]), np.array([ac_lon]), np.array([ac_alt]))[0])

    clearance_km = (-dist[k] if inside else dist[k]) * nm / 1000.   # NM -> km

    # the bearing of the nearest point relative to the heading
    bearing = degto180(ac_hdg - qdr_nearest) # relative to the aircraft heading, in [-180, 180]
    cos_bearing = float(np.cos(np.radians(bearing)))
    sin_bearing = float(np.sin(np.radians(bearing)))

    if inside:
        return float(clearance_km), cos_bearing, sin_bearing, 1.0, -1.0 # [km, -, -, -, -]

    # tangents: bearings of the vertices - bearing of the nearest point
    rel = degto180(qdr_vertices - qdr_nearest) # [deg, range -180, 180]
    qdr_left = qdr_nearest + rel.min()    # smallest bearing = left-most [deg]
    qdr_right = qdr_nearest + rel.max()   # largest bearing = right-most [deg]
    left_edge = degto180(ac_hdg - qdr_left) / 180.0 # normalized bearing to [-1, 1]
    right_edge = degto180(ac_hdg - qdr_right) / 180.0 # normalized bearing to [-1, 1]
    return float(clearance_km), cos_bearing, sin_bearing, float(left_edge), float(right_edge) # [km, -, -, -, -]
