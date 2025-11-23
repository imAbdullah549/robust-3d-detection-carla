import argparse
import time
from pathlib import Path
from collections import deque
import math

import carla
import numpy as np
from PIL import Image

from .weather_presets import WEATHER_PRESETS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate KITTI-style data from CARLA under different weather conditions."
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        required=True,
        help="Name of the scene folder, e.g. scene_dense_fog_v1",
    )
    parser.add_argument(
        "--weather",
        type=str,
        required=True,
        choices=WEATHER_PRESETS.keys(),
        help="Weather preset key, e.g. dense_fog, clear_night, rainy_night, heavy_rain_day",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=600,
        help="Number of frames to record",
    )
    return parser.parse_args()


def get_camera_intrinsic_matrix(camera_actor):
    image_w = float(camera_actor.attributes["image_size_x"])
    image_h = float(camera_actor.attributes["image_size_y"])
    fov = float(camera_actor.attributes["fov"])
    focal_length = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    cx = image_w / 2.0
    cy = image_h / 2.0
    P2 = np.array(
        [
            [focal_length, 0, cx, 0],
            [0, focal_length, cy, 0],
            [0, 0, 1, 0],
        ]
    )
    return P2


def main():
    args = parse_args()

    # === CONFIGURATION ===
    SCENE_NAME = args.scene_name
    SAVE_ROOT = Path("output") / SCENE_NAME
    IMAGE_DIR = SAVE_ROOT / "image_2"
    LIDAR_DIR = SAVE_ROOT / "velodyne"
    LABEL_DIR = SAVE_ROOT / "label_2"
    CALIB_DIR = SAVE_ROOT / "calib"
    for d in [IMAGE_DIR, LIDAR_DIR, LABEL_DIR, CALIB_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # === CARLA to KITTI Coordinate Transformation Matrices ===
    R_carla_cam_to_kitti_cam = np.array(
        [
            [0, 1, 0],   # Carla Y (right) to Kitti X (right)
            [0, 0, -1],  # Carla Z (up) to Kitti Y (down)
            [1, 0, 0],   # Carla X (forward) to Kitti Z (forward)
        ]
    )
    T_carla_to_kitti_camera_4x4 = np.eye(4)
    T_carla_to_kitti_camera_4x4[:3, :3] = R_carla_cam_to_kitti_cam

    R_carla_lidar_to_kitti_lidar = np.array(
        [
            [1, 0, 0],   # Carla X (forward) to Kitti X (forward)
            [0, -1, 0],  # Carla Y (right) to Kitti Y (left)
            [0, 0, 1],   # Carla Z (up) to Kitti Z (up)
        ]
    )
    T_carla_to_kitti_lidar_4x4 = np.eye(4)
    T_carla_to_kitti_lidar_4x4[:3, :3] = R_carla_lidar_to_kitti_lidar

    # === CONNECT TO CARLA ===
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    if "Town03" not in world.get_map().name:
        world = client.load_world("Town03")

    # Use selected weather preset
    world.set_weather(WEATHER_PRESETS[args.weather])
    for _ in range(10):
        world.tick()

    blueprint_lib = world.get_blueprint_library()

    # === SYNCHRONOUS MODE ===
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    # === SPAWN EGO VEHICLE ===
    vehicle_bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    np.random.shuffle(spawn_points)
    vehicle = None
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            print(f"[✓] Ego vehicle spawned at: {spawn_point.location}")
            break
    if vehicle is None:
        raise RuntimeError(
            "Failed to spawn ego vehicle after trying all spawn points!"
        )
    vehicle.set_autopilot(True, traffic_manager.get_port())

    # === SPAWN TRAFFIC ===
    all_vehicles = blueprint_lib.filter("vehicle.*")
    two_wheelers = [
        bp
        for bp in all_vehicles
        if bp.has_attribute("number_of_wheels")
        and int(bp.get_attribute("number_of_wheels")) == 2
    ]
    cars = [
        bp
        for bp in all_vehicles
        if bp.has_attribute("number_of_wheels")
        and int(bp.get_attribute("number_of_wheels")) > 2
    ]
    vehicles_list = []
    np.random.shuffle(spawn_points)
    for i in range(80):
        if i >= len(spawn_points):
            break
        if i % 8 == 0 and two_wheelers:
            npc_bp = np.random.choice(two_wheelers)
        else:
            npc_bp = np.random.choice(cars)
        if (
            spawn_points[i].location.distance(vehicle.get_location()) < 20.0
        ):
            continue
        npc = world.try_spawn_actor(npc_bp, spawn_points[i])
        if npc:
            npc.set_autopilot(True, traffic_manager.get_port())
            vehicles_list.append(npc)
    print(f"Spawned {len(vehicles_list)} NPC vehicles.")

    # === SPAWN PEDESTRIANS ===
    pedestrian_bps = blueprint_lib.filter("walker.pedestrian.*")
    walker_spawn_points = []
    for _ in range(100):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            walker_spawn_points.append(spawn_point)
    pedestrian_list = []
    for spawn_point in walker_spawn_points:
        ped_bp = np.random.choice(pedestrian_bps)
        if (
            spawn_point.location.distance(vehicle.get_location()) < 10.0
        ):
            continue
        ped = world.try_spawn_actor(ped_bp, spawn_point)
        if ped:
            pedestrian_list.append(ped)
    print(f"[✓] Spawned {len(pedestrian_list)} pedestrians.")

    # === SENSORS ===
    camera_bp = blueprint_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "1242")
    camera_bp.set_attribute("image_size_y", "375")
    camera_bp.set_attribute("fov", "90.0")
    camera_transform = carla.Transform(
        carla.Location(x=1.5, y=0.0, z=2.3),
        carla.Rotation(pitch=-5.0),
    )
    camera = world.spawn_actor(
        camera_bp, camera_transform, attach_to=vehicle
    )

    lidar_bp = blueprint_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", "80")
    lidar_bp.set_attribute("rotation_frequency", "10")
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("points_per_second", "1300000")
    lidar_transform = carla.Transform(carla.Location(x=0.0, z=1.73))
    lidar = world.spawn_actor(
        lidar_bp, lidar_transform, attach_to=vehicle
    )

    image_queue = deque()
    lidar_queue = deque()
    camera.listen(lambda image: image_queue.append(image))
    lidar.listen(lambda data: lidar_queue.append(data))

    def write_calib(frame, camera_actor, lidar_actor):
        T_world_lidar_carla = np.array(lidar_actor.get_transform().get_matrix())
        T_world_camera_carla = np.array(camera_actor.get_transform().get_matrix())
        P2 = get_camera_intrinsic_matrix(camera_actor)
        T_velo_cam_kitti = (
            T_carla_to_kitti_camera_4x4
            @ np.linalg.inv(T_world_camera_carla)
            @ T_world_lidar_carla
            @ np.linalg.inv(T_carla_to_kitti_lidar_4x4)
        )
        Tr_velo_to_cam = T_velo_cam_kitti[:3, :]
        Tr_imu_to_velo = np.eye(3, 4)
        calib_content = f"""P0: {' '.join([f'{x:.12e}' for x in P2.flatten()])}
P1: {' '.join([f'{x:.12e}' for x in P2.flatten()])}
P2: {' '.join([f'{x:.12e}' for x in P2.flatten()])}
P3: {' '.join([f'{x:.12e}' for x in P2.flatten()])}
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: {' '.join([f'{x:.12e}' for x in Tr_velo_to_cam.flatten()])}
Tr_imu_to_velo: {' '.join([f'{x:.12e}' for x in Tr_imu_to_velo.flatten()])}
"""
        (CALIB_DIR / f"{frame:06}.txt").write_text(calib_content)

    def save_labels(frame, camera_actor, lidar_actor, ego_vehicle):
        label_path = LABEL_DIR / f"{frame:06}.txt"
        T_world_cam_carla = np.array(camera_actor.get_transform().get_matrix())
        T_cam_world_carla = np.linalg.inv(T_world_cam_carla)
        P2 = get_camera_intrinsic_matrix(camera_actor)
        image_w = float(camera_actor.attributes["image_size_x"])
        image_h = float(camera_actor.attributes["image_size_y"])

        actors_to_label = [
            a
            for a in world.get_actors()
            if (
                ("vehicle." in a.type_id and a.id != ego_vehicle.id)
                or "walker.pedestrian" in a.type_id
            )
        ]

        label_lines = []
        has_valid_label = False

        for actor in actors_to_label:
            bp = blueprint_lib.find(actor.type_id)
            if "walker.pedestrian" in actor.type_id:
                obj_type = "Pedestrian"
            elif "bicycle" in actor.type_id or "motorcycle" in actor.type_id or (
                bp.has_attribute("number_of_wheels")
                and int(bp.get_attribute("number_of_wheels")) == 2
            ):
                obj_type = "Cyclist"
            else:
                obj_type = "Car"

            bb = actor.bounding_box
            trans = actor.get_transform()
            T_world_obj_carla = np.array(trans.get_matrix())

            # Dimensions (KITTI): height, width, length
            h = bb.extent.z * 2
            w = bb.extent.y * 2
            l = bb.extent.x * 2

            bb_local_carla = np.array(
                [bb.location.x, bb.location.y, bb.location.z, 1.0]
            )
            bb_center_world_carla = T_world_obj_carla @ bb_local_carla
            bb_center_cam_carla = T_cam_world_carla @ bb_center_world_carla
            if bb_center_cam_carla[0] < 0.1:
                continue

            bb_center_cam_kitti = (
                T_carla_to_kitti_camera_4x4 @ bb_center_cam_carla
            )
            x_cam, y_cam, z_cam = bb_center_cam_kitti[:3]
            y_cam_kitti_bottom = y_cam + h / 2.0

            # 2D bounding box
            x_bb, y_bb, z_bb = (
                bb.location.x,
                bb.location.y,
                bb.location.z,
            )
            dx, dy, dz = bb.extent.x, bb.extent.y, bb.extent.z
            corners_local_carla = np.array(
                [
                    [x_bb + dx, y_bb + dy, z_bb - dz, 1],
                    [x_bb + dx, y_bb - dy, z_bb - dz, 1],
                    [x_bb - dx, y_bb - dy, z_bb - dz, 1],
                    [x_bb - dx, y_bb + dy, z_bb - dz, 1],
                    [x_bb + dx, y_bb + dy, z_bb + dz, 1],
                    [x_bb + dx, y_bb - dy, z_bb + dz, 1],
                    [x_bb - dx, y_bb - dy, z_bb + dz, 1],
                    [x_bb - dx, y_bb + dy, z_bb + dz, 1],
                ]
            ).T

            T_cam_actor_kitti = (
                T_carla_to_kitti_camera_4x4
                @ T_cam_world_carla
                @ T_world_obj_carla
            )
            corners_cam_kitti = (
                T_cam_actor_kitti @ corners_local_carla
            ).T[:, :3]
            proj_corners = (
                P2 @ np.hstack((corners_cam_kitti, np.ones((8, 1)))).T
            ).T
            if (proj_corners[:, 2] <= 0).any():
                continue
            proj_corners[:, 0] /= proj_corners[:, 2]
            proj_corners[:, 1] /= proj_corners[:, 2]
            bbox_left, bbox_top = np.min(proj_corners[:, :2], axis=0)
            bbox_right, bbox_bottom = np.max(proj_corners[:, :2], axis=0)

            if bbox_right <= bbox_left or bbox_bottom <= bbox_top:
                continue
            if (
                bbox_right < 0
                or bbox_left > image_w - 1
                or bbox_bottom < 0
                or bbox_top > image_h - 1
            ):
                continue

            box_height_2d = bbox_bottom - bbox_top
            distance_from_ego = np.linalg.norm(
                bb_center_world_carla[:3]
                - np.array(
                    [
                        ego_vehicle.get_location().x,
                        ego_vehicle.get_location().y,
                        ego_vehicle.get_location().z,
                    ]
                )
            )

            if box_height_2d < 25 or distance_from_ego > 70:
                line = (
                    f"DontCare -1 -1 -10.00 "
                    f"{bbox_left:.2f} {bbox_top:.2f} {bbox_right:.2f} {bbox_bottom:.2f} "
                    f"-1.00 -1.00 -1.00 -1000.00 -1000.00 -1000.00 -10.00\n"
                )
                label_lines.append(line)
                continue

            has_valid_label = True
            bbox_left = np.clip(bbox_left, 0, image_w - 1)
            bbox_top = np.clip(bbox_top, 0, image_h - 1)
            bbox_right = np.clip(bbox_right, 0, image_w - 1)
            bbox_bottom = np.clip(bbox_bottom, 0, image_h - 1)

            carla_yaw_rad = math.radians(trans.rotation.yaw)
            rotation_y = -carla_yaw_rad - (math.pi / 2.0)
            rotation_y = (rotation_y + math.pi) % (2 * math.pi) - math.pi

            alpha = rotation_y - math.atan2(x_cam, z_cam)
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

            truncated = 0.0
            occluded = 0
            line = (
                f"{obj_type} {truncated:.2f} {occluded} {alpha:.2f} "
                f"{bbox_left:.2f} {bbox_top:.2f} {bbox_right:.2f} {bbox_bottom:.2f} "
                f"{h:.2f} {w:.2f} {l:.2f} {x_cam:.2f} {y_cam_kitti_bottom:.2f} {z_cam:.2f} {rotation_y:.2f}\n"
            )
            label_lines.append(line)

        if has_valid_label:
            with open(label_path, "w") as f:
                f.writelines(label_lines)
            return True
        return False

    # === MAIN RECORDING LOOP ===
    print(f"Recording {args.frames} frames of data for scene '{SCENE_NAME}' with weather '{args.weather}'...")
    frame_id = 0
    try:
        while frame_id < args.frames:
            world.tick()
            if image_queue and lidar_queue:
                image = image_queue.popleft()
                lidar_data = lidar_queue.popleft()
                if image.frame != lidar_data.frame:
                    print(
                        f"[!] Skipped frame due to mismatched sensor frames. "
                        f"Image: {image.frame}, Lidar: {lidar_data.frame}"
                    )
                    continue

                is_frame_valid = save_labels(
                    frame_id, camera, lidar, vehicle
                )
                if not is_frame_valid:
                    print(
                        f"[!] Skipped frame {frame_id} due to having no valid labels."
                    )
                    continue

                write_calib(frame_id, camera, lidar)

                arr = np.frombuffer(
                    image.raw_data, dtype=np.uint8
                ).reshape(image.height, image.width, 4)
                Image.fromarray(arr[:, :, :3]).save(
                    IMAGE_DIR / f"{frame_id:06}.png"
                )

                points_carla = np.frombuffer(
                    lidar_data.raw_data, dtype=np.float32
                ).reshape(-1, 4)
                points_carla = points_carla[points_carla[:, 0] > 0]
                points_homogeneous_carla = np.hstack(
                    (points_carla[:, :3], np.ones((points_carla.shape[0], 1)))
                )
                points_kitti_homogeneous = (
                    T_carla_to_kitti_lidar_4x4
                    @ points_homogeneous_carla.T
                ).T
                points_kitti_final = np.hstack(
                    (points_kitti_homogeneous[:, :3], points_carla[:, 3:])
                )
                points_kitti_final.astype(np.float32).tofile(
                    LIDAR_DIR / f"{frame_id:06}.bin"
                )

                print(f"[✓] Saved frame {frame_id}")
                frame_id += 1
    except Exception as e:
        print(f"\n[✗] An error occurred in the main loop: {e}")
    finally:
        print("\nCleaning up actors and sensors...")
        try:
            if "camera" in locals() and camera.is_alive:
                camera.destroy()
            if "lidar" in locals() and lidar.is_alive:
                lidar.destroy()
            if "vehicle" in locals() and vehicle.is_alive:
                vehicle.destroy()
            for v in vehicles_list:
                if v.is_alive:
                    v.destroy()
            for p in pedestrian_list:
                if p.is_alive:
                    p.destroy()
        except Exception as e:
            print(f"Error during cleanup: {e}")

        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print(f"\n✅ Recording complete. Data saved in: {SAVE_ROOT}")


if __name__ == "__main__":
    main()
