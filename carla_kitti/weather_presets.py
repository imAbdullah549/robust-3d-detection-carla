# carla_kitti/weather_presets.py
import carla

WEATHER_PRESETS = {
    "clear_day": carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_altitude_angle=60.0,   
        fog_density=0.0,
        fog_distance=1000.0,
        wetness=0.0,
    ),
    "dense_fog": carla.WeatherParameters(
        cloudiness=50.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=5.0, sun_altitude_angle=50.0,
        fog_density=100.0, fog_distance=5.0, wetness=0.0,
    ),
    "clear_night": carla.WeatherParameters(
        cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=5.0, sun_altitude_angle=-10.0,
        fog_density=0.0, fog_distance=1000.0, wetness=0.0,
    ),
    "rainy_night": carla.WeatherParameters(
        cloudiness=90.0, precipitation=70.0, precipitation_deposits=80.0,
        wind_intensity=50.0, sun_altitude_angle=-5.0,
        fog_density=20.0, fog_distance=60.0, wetness=100.0,
    ),
    "heavy_rain_day": carla.WeatherParameters(
        cloudiness=100.0, precipitation=100.0, precipitation_deposits=90.0,
        wind_intensity=70.0, sun_altitude_angle=45.0,
        fog_density=15.0, fog_distance=80.0, wetness=100.0,
    ),
}
