# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading

from pyzbar import pyzbar

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True

current_qr_shelf_id, past_qr_shelf_id = 0,0
qr_angle = 0.0

class WindowProgressTable:
	def __init__(self, root, shelf_count):
		self.root = root
		self.root.title("Shelf Objects & QR Link")
		self.root.attributes("-topmost", True)

		self.row_count = 2
		self.col_count = shelf_count

		self.boxes = []
		for row in range(self.row_count):
			row_boxes = []
			for col in range(self.col_count):
				box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
						  relief="solid", font=("Helvetica", 14))
				box.insert(tk.END, "NULL")
				box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
				row_boxes.append(box)
			self.boxes.append(row_boxes)

		# Make the grid layout responsive.
		for row in range(self.row_count):
			self.root.grid_rowconfigure(row, weight=1)
		for col in range(self.col_count):
			self.root.grid_columnconfigure(col, weight=1)

	def change_box_color(self, row, col, color):
		self.boxes[row][col].config(bg=color)

	def change_box_text(self, row, col, text):
		self.boxes[row][col].delete(1.0, tk.END)
		self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
	global box_app
	root = tk.Tk()
	box_app = WindowProgressTable(root, shelf_count)
	root.mainloop()


class WarehouseExplore(Node):
	""" Initializes warehouse explorer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('warehouse_explore')

		self.action_client = ActionClient(
			self,
			NavigateToPose,
			'/navigate_to_pose')

		self.subscription_pose = self.create_subscription(
			PoseWithCovarianceStamped,
			'/pose',
			self.pose_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_global_map = self.create_subscription(
			OccupancyGrid,
			'/global_costmap/costmap',
			self.global_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_simple_map = self.create_subscription(
			OccupancyGrid,
			'/map',
			self.simple_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_status = self.create_subscription(
			Status,
			'/cerebri/out/status',
			self.cerebri_status_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_behavior = self.create_subscription(
			BehaviorTreeLog,
			'/behavior_tree_log',
			self.behavior_tree_log_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_shelf_objects = self.create_subscription(
			WarehouseShelf,
			'/shelf_objects',
			self.shelf_objects_callback,
			QOS_PROFILE_DEFAULT)

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		self.publisher_joy = self.create_publisher(
			Joy,
			'/cerebri/in/joy',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_qr_decode = self.create_publisher(
			CompressedImage,
			"/debug_images/qr_code",
			QOS_PROFILE_DEFAULT)

		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			"/shelf_data",
			QOS_PROFILE_DEFAULT)

		self.declare_parameter('shelf_count', 1)
		self.declare_parameter('initial_angle', 0.0)

		self.shelf_count = \
			self.get_parameter('shelf_count').get_parameter_value().integer_value
		self.initial_angle = \
			self.get_parameter('initial_angle').get_parameter_value().double_value

		# --- Robot State ---
		self.armed = False
		self.logger = self.get_logger()

		# --- Robot Pose ---
		self.pose_curr = PoseWithCovarianceStamped()
		self.buggy_pose_x = 0.0
		self.buggy_pose_y = 0.0
		self.buggy_center = (0.0, 0.0)
		self.world_center = (0.0, 0.0)

		# --- Map Data ---
		self.simple_map_curr = None
		self.global_map_curr = None

		# --- Goal Management ---
		self.xy_goal_tolerance = 0.5
		self.goal_completed = True  # No goal is currently in-progress.
		self.goal_handle_curr = None
		self.cancelling_goal = False
		self.recovery_threshold = 10

		# --- Goal Creation ---
		self._frame_id = "map"

		# --- Exploration Parameters ---
		self.max_step_dist_world_meters = 7.0
		self.min_step_dist_world_meters = 4.0
		self.full_map_explored_count = 0

		# --- QR Code Data ---
		self.qr_code_str = "Empty"
		if PROGRESS_TABLE_GUI:
			self.table_row_count = 0
			self.table_col_count = 0

		# --- Shelf Data ---
		self.shelf_objects_curr = WarehouseShelf()
		self.temp_shelves = []  # Temporary storage for detected shelves.
		self.shelves = []

		self.curr_shelf_details = None #details of the current shelf.
		self.candidate_shelves = [] #list of all eligible candidate shelves
		self.global_navigation_done = False

		self.pending_goals = []  # List to store pending goals for exploration.
		self.goal_in_progress = False  # Flag to indicate if a goal is currently in progress.

		self.qr_frame_counter = 0 # Process every 10th frame for qr.
		self.qr_shelf_id_decoded = 0 #Access the current shelf's id by using this global variable
		self.qr_angle_decoded = 0.0 #Access the next shelf's angle by using this global variable

		self.shelf_num = 0 # Shelf number for the current shelf being processed.

		self.on_last_shelf = False # Flag to indicate if the rover is on the last shelf.


	def pose_callback(self, message):
		"""Callback function to handle pose updates.

		Args:
			message: ROS2 message containing the current pose of the rover.

		Returns:
			None
		"""
		self.pose_curr = message
		self.buggy_pose_x = message.pose.pose.position.x
		self.buggy_pose_y = message.pose.pose.position.y
		self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)

	def simple_map_callback(self, message):
		"""Callback function to handle simple map updates.

		Args:
			message: ROS2 message containing the simple map data.

		Returns:
			None
		"""
		self.simple_map_curr = message
		map_info = self.simple_map_curr.info
		self.world_center = self.get_world_coord_from_map_coord(
			map_info.width / 2, map_info.height / 2, map_info
		)

		height = map_info.height
		width = map_info.width

		# Reshape to [height, width] (y, x)
		map_array = np.array(message.data).reshape((height, width))
		obstacle_mask = (map_array == 100).astype(np.uint8) * 255
		contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		MIN_WIDTH, MAX_WIDTH = 26, 35
		MIN_HEIGHT, MAX_HEIGHT = 8, 12

		for idx, cnt in enumerate(contours):
			rect = cv2.minAreaRect(cnt)
			(cx, cy), (w, h), _ = rect

			width = max(w, h)
			height_ = min(w, h)

			if not (MIN_WIDTH <= width <= MAX_WIDTH and MIN_HEIGHT <= height_ <= MAX_HEIGHT):
				continue

			box = cv2.boxPoints(rect).astype(np.float32)

			# Find the longest edge for orientation
			edges = []
			for i in range(4):
				pt1 = box[i]
				pt2 = box[(i + 1) % 4]
				edge_len = np.linalg.norm(pt2 - pt1)
				edges.append((edge_len, pt1, pt2))

			edges.sort(key=lambda x: -x[0])
			pt1, pt2 = edges[0][1], edges[0][2]

			# Consistent direction
			if pt1[1] > pt2[1]:
				start, end = pt1, pt2
			elif pt2[1] > pt1[1]:
				start, end = pt2, pt1
			else:
				start, end = (pt1, pt2) if pt1[0] < pt2[0] else (pt2, pt1)

			# Get yaw from the longest edge vector
			yaw = self.create_yaw_from_vector(end[0], end[1], start[0], start[1])
			if yaw < 0:
				yaw += math.pi

			# Convert center to world coordinates
			center_x_world, center_y_world = self.get_world_coord_from_map_coord(cx, cy, map_info)

			# === Compute scanning points ===

			# QR scanner positions (along ±yaw direction) - short sides
			qr_dx = 40 * math.cos(yaw)
			qr_dy = 40 * math.sin(yaw)
			qr_x1 = center_x_world + qr_dx * map_info.resolution
			qr_y1 = center_y_world + qr_dy * map_info.resolution
			qr_x2 = center_x_world - qr_dx * map_info.resolution
			qr_y2 = center_y_world - qr_dy * map_info.resolution

			# Object scanner positions (along ±(yaw + 90°)) - long sides
			obj_dx = 55 * math.cos(yaw + math.pi / 2)
			obj_dy = 55 * math.sin(yaw + math.pi / 2)
			obj_x1 = center_x_world + obj_dx * map_info.resolution
			obj_y1 = center_y_world + obj_dy * map_info.resolution
			obj_x2 = center_x_world - obj_dx * map_info.resolution
			obj_y2 = center_y_world - obj_dy * map_info.resolution

			# Yaw orientations for scanning
			qr_scan_yaw_1 = yaw + math.pi         # Facing toward the shelf
			qr_scan_yaw_2 = yaw                   # Opposite side

			obj_scan_yaw_1 = yaw - math.pi / 2    # Facing perpendicular to shelf
			obj_scan_yaw_2 = yaw + math.pi / 2    # Opposite side

			# Append shelf data with all scanning points
			shelf = {
				'center_x': center_x_world,
				'center_y': center_y_world,
				'width': width * map_info.resolution,
				'height': height_ * map_info.resolution,
				'yaw': yaw,

				'qr_scans': [
					{'x': qr_x1, 'y': qr_y1, 'yaw': qr_scan_yaw_1},
					{'x': qr_x2, 'y': qr_y2, 'yaw': qr_scan_yaw_2},
				],
				'obj_scans': [
					{'x': obj_x1, 'y': obj_y1, 'yaw': obj_scan_yaw_1},
					{'x': obj_x2, 'y': obj_y2, 'yaw': obj_scan_yaw_2},
				],

				'visited': False,
			}
			if len(self.temp_shelves) == 0:
				self.temp_shelves.append(shelf)
			else:
				self.add_shelf_if_unique(shelf)
			# self.get_logger().info(f"Temp shelves = ")
			# for s in self.temp_shelves:
			# 	self.get_logger().info(f"{s['center_x']}, {s['center_y']}, {s['yaw']}")

			# self.get_logger().info(
			# 	f"Detected shelf {idx + 1}: Center=({center_x_world:.2f}, {center_y_world:.2f}), "
			# )
			self.shelves = self.temp_shelves

	def add_shelf_if_unique(self, shelf):
		for i, s in enumerate(self.temp_shelves):
			if math.hypot(shelf['center_x'] - s['center_x'],shelf['center_y'] - s['center_y']) < 0.5:
				self.temp_shelves[i] = shelf
				return
		self.temp_shelves.append(shelf)


	def global_map_callback(self, message):
		"""Callback function to handle global map updates.

		Args:
			message: ROS2 message containing the global map data.

		Returns:
			None
		"""
		self.global_map_curr = message
		# FOR MANUAL MODE ADDED REURN STATEMENT
		# return

		if not self.goal_completed:
			return

		height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
		map_array = np.array(self.global_map_curr.data).reshape((height, width))

		

		map_info = self.global_map_curr.info

		if (len(self.pending_goals) != 0):
			if(len(self.pending_goals) == 1):
				time.sleep(3)  # Wait for the rover to reach the goal.
			self._send_next_goal()

		else:

			if self.shelf_num==0:
				angle = self.initial_angle
				point = (0,0)
			else:
				angle = self.qr_angle_decoded
				self.logger.info(f"Angle for shelf {self.shelf_num+1} is {angle}")
				prev_shelf = self.curr_shelf_details
				point = (prev_shelf['center_x'], prev_shelf['center_y'])

			self.logger.info(f"self.shelves =  {self.shelves}")
			self.candidate_shelves = self.find_shelves_in_direction(point, angle)


			if len(self.candidate_shelves) >= 1:
				shelf = self.candidate_shelves[0]
				self.min_dist_goals()
				qr_goal = self.create_goal_from_world_coord(
						self.candidate_shelves[0]['qr_scan'][0][0], self.candidate_shelves[0]['qr_scan'][0][1], self.candidate_shelves[0]['qr_scan'][0][2])
				self.pending_goals.append(qr_goal)
				obj_goal = self.create_goal_from_world_coord(
						self.candidate_shelves[0]['obj_scan'][0][0], self.candidate_shelves[0]['obj_scan'][0][1], self.candidate_shelves[0]['obj_scan'][0][2])
				self.pending_goals.append(obj_goal)
				for s in self.shelves:
					if s['center_x'] == shelf['center_x'] and s['center_y'] == shelf['center_y']:
						s['visited'] = True
						break
				self.shelf_num += 1
				shelf['shelf_num'] = self.shelf_num
				self.curr_shelf_details = shelf
				self._send_next_goal()
				self.logger.info(f"Detected objects: {self.shelf_objects_curr.object_name}, Counts: {self.shelf_objects_curr.object_count}")

				
				if self.shelf_objects_curr.object_name and self.shelf_objects_curr.object_count:
					shelf_data_message = WarehouseShelf()
					print(f"Detected objects: {self.shelf_objects_curr.object_name}, Counts: {self.shelf_objects_curr.object_count}")
					shelf_data_message.object_name = self.shelf_objects_curr.object_name
					shelf_data_message.object_count = self.shelf_objects_curr.object_count
					shelf_data_message.qr_decoded = self.qr_code_str
					self.publisher_shelf_data.publish(shelf_data_message)
			
			else:
				# self.get_logger().info(f"Candidate shelves in given direction : {self.candidate_shelves}")
				frontiers = self.get_frontiers_for_space_exploration(map_array)
				if frontiers:
					closest_frontier = None
					min_distance_curr = float('inf')

					for fy, fx in frontiers:
						fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy,
													map_info)
						# distance = euclidean((fx_world, fy_world), self.buggy_center)
						# if (distance < min_distance_curr and
						# 	distance <= self.max_step_dist_world_meters and
						# 	distance >= self.min_step_dist_world_meters):
						# 	min_distance_curr = distance
						# 	closest_frontier = (fy, fx)

						angle_rad = math.radians(angle)
						dx_guidance = math.cos(angle_rad)
						dy_guidance = math.sin(angle_rad)
						
						# Vector from robot to frontier
						vx_robot_to_frontier = fx_world - self.buggy_center[0]
						vy_robot_to_frontier = fy_world - self.buggy_center[1]
						
						# Check if frontier is in forward direction from robot's current position
						dot_product_robot = vx_robot_to_frontier * dx_guidance + vy_robot_to_frontier * dy_guidance
						if dot_product_robot < 0:
							continue

						perp_distance = self._distance_point_to_line(fx_world, fy_world,point[0], point[1], angle)
						robot_distance = euclidean((fx_world, fy_world), self.buggy_center)
						if (perp_distance < min_distance_curr and 
							robot_distance <= self.max_step_dist_world_meters and
							robot_distance >= self.min_step_dist_world_meters):
							min_distance_curr = perp_distance
							closest_frontier = (fy, fx)		

					if closest_frontier:
						fy, fx = closest_frontier
						goal = self.create_goal_from_map_coord(fx, fy, map_info)
						self.send_goal_from_world_pose(goal)
						self.get_logger().info("Sending goal for space exploration.")
						return
					else:
						self.max_step_dist_world_meters += 2.0
						new_min_step_dist = self.min_step_dist_world_meters - 1.0
						self.min_step_dist_world_meters = max(0.25, new_min_step_dist)

				else:
					# self.full_map_explored_count += 1
					self.get_logger().info(f"Nothing found in frontiers; count = {self.full_map_explored_count}")
					self.global_code_navigation(message)
						

			return

	def _distance_point_to_line(self, px, py, x0, y0, angle):
		"""
		Calculates the perpendicular distance from a point (px, py) to a ray
		defined by a point (x0, y0) and an angle in radians.
		Only considers points that are in the forward direction of the ray.
		"""
		angle = math.radians(angle)  # Ensure angle is in radians
		
		# Direction vector of the ray
		dx = math.cos(angle)
		dy = math.sin(angle)

		# Vector from the ray's origin to the point
		vx = px - x0
		vy = py - y0

		# Project the point vector onto the ray direction
		dot_product = vx * dx + vy * dy
		
		# If dot product is negative, the point is behind the ray origin
		if dot_product < 0:
			return float('inf')  # Return a very large distance to exclude this point
		
		# Calculate perpendicular distance (same as before)
		distance = abs(vx * dy - vy * dx)
		return distance


	def distance(self, p1, p2):
		return math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])

	def min_dist_goals(self):
		buggy_pos = {'x': self.buggy_center[0], 'y': self.buggy_center[1]}

		for shelf in self.candidate_shelves:
			# Sort QR scans by distance from buggy
			shelf['qr_scans'].sort(key=lambda scan: self.distance(scan, buggy_pos))

			# Sort Object scans by distance from buggy
			shelf['obj_scans'].sort(key=lambda scan: self.distance(scan, buggy_pos))

	
	def find_shelves_in_direction(self,current_point, yaw_angle, max_distance=999999, angle_tolerance_deg=25): # Can change tolerance degree for more selctive shelf identification.
		matched = []
		
		angle_tolerance_rad = math.radians(angle_tolerance_deg)
		yaw_angle_rad = math.radians(yaw_angle)

		for shelf in self.shelves:
			dx = shelf['center_x'] - current_point[0]
			dy = shelf['center_y'] - current_point[1]
			distance = math.hypot(dx, dy)
			if distance > max_distance:
				continue

			angle_to_shelf = math.atan2(dy, dx)
			angle_diff = abs((angle_to_shelf - yaw_angle_rad + math.pi) % (2 * math.pi) - math.pi)

			if angle_diff <= angle_tolerance_rad:
				shelf_copy = dict(shelf)
				shelf_copy['distance'] = distance
				matched.append(shelf_copy)

		matched.sort(key=lambda s: s['distance'])
		matched = [s for s in matched if not s['visited']]
		return matched
	

	def global_code_navigation(self,message):
		"""Global navigation function to handle complete map exploration."""
		self.get_logger().info("Exploration complete. Navigating to the next shelf.")

		self.shelves = self.select_final_shelves(self.temp_shelves, self.shelf_count)
		self.get_logger().info(f"Shelves list: {self.shelves}")
		count = self.shelf_count

	
		if (len(self.pending_goals) != 0):
			# self.logger.info(f"if loop:{len(self.pending_goals)}")
			self._send_next_goal()
			if self.shelf_num==count:
				self.on_last_shelf = True
				pass
		else:
			self.logger.info(f"else loop:{len(self.pending_goals)}")
			if self.shelf_num==0:
				angle = self.initial_angle
				point = (0,0)
			else:
				angle = self.qr_angle_decoded
				self.logger.info(f"Angle for shelf {self.shelf_num+1} is {angle}")
				prev_shelf = self.curr_shelf_details
				point = (prev_shelf['center_x'], prev_shelf['center_y'])
				self.logger.info(f"Point for shelf {self.shelf_num} is {point}")

			self.candidate_shelves = self.find_shelves_in_direction(point, angle)

			self.get_logger().info(f"Candidate shelves in given direction : {self.candidate_shelves}")
			self.get_logger().info(f"Number of candidate shelves found: {len(self.candidate_shelves)}")

			# if len(self.candidate_shelves)>1:
			# 	pass
			if len(self.candidate_shelves) >= 1 and self.goal_completed==True:
				shelf = self.candidate_shelves[0]
				for s in self.shelves:
					if s['center_x'] == shelf['center_x'] and s['center_y'] == shelf['center_y']:
						s['visited'] = True
						break
				
				qr_goal = self.create_goal_from_world_coord(
					self.candidate_shelves[0]['qr_scan'][0], self.candidate_shelves[0]['qr_scan'][1], self.candidate_shelves[0]['qr_scan_yaw'])
				self.pending_goals.append(qr_goal)
				obj_goal = self.create_goal_from_world_coord(
					self.candidate_shelves[0]['obj_scan'][0], self.candidate_shelves[0]['obj_scan'][1], self.candidate_shelves[0]['obj_scan_yaw'])
				self.pending_goals.append(obj_goal)
				self.shelf_num += 1
				shelf['shelf_num'] = self.shelf_num
				self.curr_shelf_details = shelf
				
				# self.logger.info(f"Current pose: {self.buggy_pose_x}, {self.buggy_pose_y}")
				self._send_next_goal()
				if self.shelf_objects_curr.object_name and self.shelf_objects_curr.object_count:
					shelf_data_message = WarehouseShelf()
					print(f"Detected objects: {self.shelf_objects_curr.object_name}, Counts: {self.shelf_objects_curr.object_count}")
					shelf_data_message.object_name = self.shelf_objects_curr.object_name
					shelf_data_message.object_count = self.shelf_objects_curr.object_count
					shelf_data_message.qr_decoded = self.qr_code_str
					self.publisher_shelf_data.publish(shelf_data_message)
				# while self.buggy_pose_x != self.candidate_shelves[0]['qr_scan'][0] and self.buggy_pose_y != self.candidate_shelves[0]['qr_scan'][1]:
				# 	self.logger.info(f"Waiting for rover to reach the object scan point at {self.candidate_shelves[0]['qr_scan']}")
				# 	time.sleep(2)  
					# pass

				
				self.logger.info(f"Sending goals for shelf {self.shelf_num} with QR scan at {shelf['qr_scan']} and object scan at {shelf['obj_scan']}")
			else:
				if self.shelf_objects_curr.object_name and self.shelf_objects_curr.object_count:
					shelf_data_message = WarehouseShelf()
					print(f"Detected objects: {self.shelf_objects_curr.object_name}, Counts: {self.shelf_objects_curr.object_count}")
					shelf_data_message.object_name = self.shelf_objects_curr.object_name
					shelf_data_message.object_count = self.shelf_objects_curr.object_count
					shelf_data_message.qr_decoded = self.qr_code_str
					self.publisher_shelf_data.publish(shelf_data_message)
				self.get_logger().info(f"No candidate shelves found")


	def recovery_logic(self):
	   """Recovery logic to handle QR code scanning and shelf selection."""
	   global current_qr_shelf_id, past_qr_shelf_id, qr_angle
	   candidate_shelves = self.candidate_shelves
	   if not candidate_shelves:
		   self.get_logger().info("No candidate shelves found for recovery.")
		   return
	   if current_qr_shelf_id == past_qr_shelf_id+1:  #We're on the correct shelf.
		   past_qr_shelf_id = current_qr_shelf_id
		   #Add what you want to do when the shelf is correct.
	   elif current_qr_shelf_id != past_qr_shelf_id+1:  #We're on the wrong shelf.
		   #Move to the next shelf in the candidate_shelves list.
		   pass


	def _send_next_goal(self):
		"""Checks the queue and sends the next goal if available."""
		if self.pending_goals and not self.goal_in_progress:
			self.goal_in_progress = True
			next_goal = self.pending_goals.pop(0) # Get the next goal
			# self.logger.info(f"Inside _send_next_goal goal: {next_goal.pose.position.x}, {next_goal.pose.position.y}, {next_goal.pose.orientation.z}")
			self.send_goal_from_world_pose(next_goal)
			


	def get_frontiers_for_space_exploration(self, map_array):
		"""Identifies frontiers for space exploration.

		Args:
			map_array: 2D numpy array representing the map.

		Returns:
			frontiers: List of tuples representing frontier coordinates.
		"""
		frontiers = []
		for y in range(1, map_array.shape[0] - 1):
			for x in range(1, map_array.shape[1] - 1):
				if map_array[y, x] == -1:  # Unknown space and not visited.
					neighbors_complete = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
						(y - 1, x - 1),
						(y + 1, x - 1),
						(y - 1, x + 1),
						(y + 1, x + 1)
					]

					near_obstacle = False
					for ny, nx in neighbors_complete:
						if map_array[ny, nx] > 0:  # Obstacles.
							near_obstacle = True
							break
					if near_obstacle:
						continue

					neighbors_cardinal = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
					]

					for ny, nx in neighbors_cardinal:
						if map_array[ny, nx] == 0:  # Free space.
							frontiers.append((ny, nx))
							break

		return frontiers



	def publish_debug_image(self, publisher, image):
		"""Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: Image given by an n-dimensional numpy array.

		Returns:
			None
		"""
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)

	def qr_scan_function(self, image):
	   "Function to handle QR code scanning from camera images."""
	   # Initialize QR code detector
	   qr_codes = pyzbar.decode(image)
	   global  qr_angle, current_qr_shelf_id
	   if qr_codes:
		   for qr in qr_codes:
			   qr_data = qr.data.decode('utf-8')
			   self.qr_code_str = qr_data
			   self.get_logger().info(f"Detected QR: {qr_data}")
			   current_qr_shelf_id = qr_data[0]
			   self.qr_shelf_id_decoded = int(qr_data[0])
			   # Check if the QR code is valid
			   # recovery_logic() --> Calling recovery logic here.
			   qr_angle = qr_data[2:7]
			   self.qr_angle_decoded = float(qr_data[2:7])
			   # Draw rectangle and text on the image for debug
			   (x, y, w, h) = qr.rect
			   cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 2)
		
	def camera_image_callback(self, message):
		"""Callback function to handle incoming camera images.

		Args:
			message: ROS2 message of the type sensor_msgs.msg.CompressedImage.

		Returns:
			None
		"""
		np_arr = np.frombuffer(message.data, np.uint8)
		image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		# Process the image from front camera as needed.

		self.qr_frame_counter += 1
		if self.qr_frame_counter % 10 == 0: #Process every 10th frame.
			self.qr_scan_function(image)

		# Optional line for visualizing image on foxglove.
		self.publish_debug_image(self.publisher_qr_decode, image)



	def cerebri_status_callback(self, message):
		"""Callback function to handle cerebri status updates.

		Args:
			message: ROS2 message containing cerebri status.

		Returns:
			None
		"""
		if message.mode == 3 and message.arming == 2:
			self.armed = True
		else:
			# Initialize and arm the CMD_VEL mode.
			msg = Joy()
			msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
			msg.axes = [0.0, 0.0, 0.0, 0.0]
			# COMMENTED BOTTOM LINE FOR MANUAL MODE
			self.publisher_joy.publish(msg)

	def behavior_tree_log_callback(self, message):
		"""Alternative method for checking goal status.

		Args:
			message: ROS2 message containing behavior tree log.

		Returns:
			None
		"""
		for event in message.event_log:
			if (event.node_name == "FollowPath" and
				event.previous_status == "SUCCESS" and
				event.current_status == "IDLE"):
				# self.goal_completed = True
				# self.goal_handle_curr = None
				pass

	def shelf_objects_callback(self, message):
		"""Callback function to handle shelf objects updates.

		Args:
			message: ROS2 message containing shelf objects data.

		Returns:
			None
		"""
		self.shelf_objects_curr = message 
		# Process the shelf objects as needed.
		# How to send WarehouseShelf messages for evaluation.

		# # * Example for sending WarehouseShelf messages for evaluation.
		# if message.object_name and message.object_count and self.on_last_shelf:
		#     shelf_data_message = WarehouseShelf()
		#     print(f"Detected objects: {message.object_name}, Counts: {message.object_count}")
		#     shelf_data_message.object_name = message.object_name
		#     shelf_data_message.object_count = message.object_count
		#     shelf_data_message.qr_decoded = self.qr_code_str
			
		# # 	self.publisher_shelf_data.publish(shelf_data_message)
		"""
		* Alternatively, you may store the QR for current shelf as self.qr_code_str.
			Then, add it as self.shelf_objects_curr.qr_decoded = self.qr_code_str
			Then, publish as self.publisher_shelf_data.publish(self.shelf_objects_curr)
			This, will publish the current detected objects with the last QR decoded.
		"""

		# Optional code for populating TABLE GUI with detected objects and QR data.
		
		# if PROGRESS_TABLE_GUI:
		#     shelf = self.shelf_objects_curr
		#     obj_str = ""
		#     for name, count in zip(shelf.object_name, shelf.object_count):
		#         obj_str += f"{name}: {count}\n"

		#     box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
		#     box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
		#     self.table_row_count += 1

		#     box_app.change_box_text(self.table_row_count, self.table_col_count, self.qr_code_str)
		#     box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
		#     self.table_row_count = 0
		#     self.table_col_count += 1


	def rover_move_manual_mode(self, speed, turn):
		"""Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: The speed of the car in float. Range = [-1.0, +1.0];
				   Direction: forward for positive, reverse for negative.
			turn: Steer value of the car in float. Range = [-1.0, +1.0];
				  Direction: left turn for positive, right turn for negative.

		Returns:
			None
		"""
		msg = Joy()
		msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
		msg.axes = [0.0, speed, 0.0, turn]
		self.publisher_joy.publish(msg)


	def cancel_goal_callback(self, future):
		"""
		Callback function executed after a cancellation request is processed.

		Args:
			future (rclpy.Future): The future is the result of the cancellation request.
		"""
		cancel_result = future.result()
		if cancel_result:
			self.logger.info("Goal cancellation successful.")
			self.cancelling_goal = False  # Mark cancellation as completed (success).
			return True
		else:
			self.logger.error("Goal cancellation failed.")
			self.cancelling_goal = False  # Mark cancellation as completed (failed).
			return False

	def cancel_current_goal(self):
		"""Requests cancellation of the currently active navigation goal."""
		if self.goal_handle_curr is not None and not self.cancelling_goal:
			self.cancelling_goal = True  # Mark cancellation in-progress.
			self.logger.info("Requesting cancellation of current goal...")
			cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
			cancel_future.add_done_callback(self.cancel_goal_callback)

	def goal_result_callback(self, future):
		"""
		Callback function executed when the navigation goal reaches a final result.

		Args:
			future (rclpy.Future): The future that is result of the navigation action.
		"""
		status = future.result().status
		# NOTE: Refer https://docs.ros2.org/foxy/api/action_msgs/msg/GoalStatus.html.

		if status == GoalStatus.STATUS_SUCCEEDED:
			self.logger.info("Goal completed successfully!")
		else:
			self.logger.warn(f"Goal failed with status: {status}")

		self.goal_completed = True  # Mark goal as completed.
		self.goal_handle_curr = None  # Clear goal handle.

		self.goal_in_progress = False  # Reset goal in progress flag.
		# self.logger.info("Inside goal result callback")
		self._send_next_goal()  # Check and send the next goal if available.

	def goal_response_callback(self, future):
		"""
		Callback function executed after the goal is sent to the action server.

		Args:
			future (rclpy.Future): The future that is server's response to goal request.
		"""
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.logger.warn('Goal rejected :(')
			self.goal_completed = True  # Mark goal as completed (rejected).
			self.goal_handle_curr = None  # Clear goal handle.
		else:
			self.logger.info('Goal accepted :)')
			self.goal_completed = False  # Mark goal as in progress.
			self.goal_handle_curr = goal_handle  # Store goal handle.

			get_result_future = goal_handle.get_result_async()
			# self.logger.info(f"Inside goal reponse callback: {goal_handle.goal_id.uuid}")
			get_result_future.add_done_callback(self.goal_result_callback)

	def goal_feedback_callback(self, msg):
		"""
		Callback function to receive feedback from the navigation action.

		Args:
			msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
		"""
		distance_remaining = msg.feedback.distance_remaining
		number_of_recoveries = msg.feedback.number_of_recoveries
		navigation_time = msg.feedback.navigation_time.sec
		estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

		self.logger.debug(f"Recoveries: {number_of_recoveries}, "
				  f"Navigation time: {navigation_time}s, "
				  f"Distance remaining: {distance_remaining:.2f}, "
				  f"Estimated time remaining: {estimated_time_remaining}s")

		if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
			self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
			self.cancel_current_goal()  # Unblock by discarding the current goal.

	def send_goal_from_world_pose(self, goal_pose):
		"""
		Sends a navigation goal to the Nav2 action server.

		Args:
			goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

		Returns:
			bool: True if the goal was successfully sent, False otherwise.
		"""
		if not self.goal_completed or self.goal_handle_curr is not None:
			return False

		self.goal_completed = False  # Starting a new goal.

		goal = NavigateToPose.Goal()
		goal.pose = goal_pose

		if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
			self.logger.error('NavigateToPose action server not available!')
			return False

		# Send goal asynchronously (non-blocking).
		goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
		# self.logger.info(f"Inside send_goal_from_world_pose: {goal_pose.pose.position.x}, {goal_pose.pose.position.y}, {goal_pose.pose.orientation.z}")
		goal_future.add_done_callback(self.goal_response_callback)

		return True



	def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
		"""Helper function to get map origin and resolution."""
		if map_info:
			origin = map_info.origin
			resolution = map_info.resolution
			return resolution, origin.position.x, origin.position.y
		else:
			return None

	def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
					   -> Tuple[float, float]:
		"""Converts map coordinates to world coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			world_x = (map_x + 0.5) * resolution + origin_x
			world_y = (map_y + 0.5) * resolution + origin_y
			return (world_x, world_y)
		else:
			return (0.0, 0.0)

	def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
					   -> Tuple[int, int]:
		"""Converts world coordinates to map coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			map_x = int((world_x - origin_x) / resolution)
			map_y = int((world_y - origin_y) / resolution)
			return (map_x, map_y)
		else:
			return (0, 0)

	def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
		"""Helper function to create a Quaternion from a yaw angle."""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		q = Quaternion()
		q.x = 0.0
		q.y = 0.0
		q.z = sy
		q.w = cy
		return q

	def create_yaw_from_vector(self, dest_x: float, dest_y: float,
				   source_x: float, source_y: float) -> float:
		"""Calculates the yaw angle from a source to a destination point.
			NOTE: This function is independent of the type of map used.

			Input: World coordinates for destination and source.
			Output: Angle (in radians) with respect to x-axis.
		"""
		delta_x = dest_x - source_x
		delta_y = dest_y - source_y
		yaw = math.atan2(delta_y, delta_x)

		return yaw

	def create_goal_from_world_coord(self, world_x: float, world_y: float,
					 yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from world coordinates.
			NOTE: This function is independent of the type of map used.
		"""
		goal_pose = PoseStamped()
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.header.frame_id = self._frame_id

		goal_pose.pose.position.x = world_x
		goal_pose.pose.position.y = world_y

		if yaw is None and self.pose_curr is not None:
			# Calculate yaw from current position to goal position.
			source_x = self.pose_curr.pose.pose.position.x
			source_y = self.pose_curr.pose.pose.position.y
			yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
		elif yaw is None:
			yaw = 0.0
		else:  # No processing needed; yaw is supplied by the user.
			pass

		goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

		pose = goal_pose.pose.position
		print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
		return goal_pose

	def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
					   yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from map coordinates."""
		world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

		return self.create_goal_from_world_coord(world_x, world_y, yaw)
	





def main(args=None):
	rclpy.init(args=args)

	warehouse_explore = WarehouseExplore()

	if PROGRESS_TABLE_GUI:
		gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
		gui_thread.start()

	rclpy.spin(warehouse_explore)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	warehouse_explore.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
