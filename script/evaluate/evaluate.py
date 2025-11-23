#!/usr/bin/env python3
"""ROS node that converts pose/odometry messages into evo-compatible TUM files."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import rospy
from roslib.message import get_message_class

# openvins config
CONFIG_PATH = Path(__file__).resolve().with_name("openvins_config.json")

# orb-slam2 config
# CONFIG_PATH = Path(__file__).resolve().with_name("orb_slam2_config.json")

PoseRecord = Tuple[float, Tuple[float, float, float], Tuple[float, float, float, float]]
PoseExtractor = Callable[[Any], PoseRecord]


def load_config(path: Path) -> Dict[str, Any]:
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {path}")

	with path.open("r", encoding="utf-8") as fp:
		data = json.load(fp)

	defaults = {
		"topic_name": "/vio_backend/odometry",
		"message_type": "nav_msgs/Odometry",
		"output_directory": "~/ws/catkin_ws/src/vio_backend/log/evo",
		"dataset": "default_dataset",
		"file_prefix": "odom_export",
		"queue_size": 100,
	}

	defaults.update(data or {})
	return defaults


def _timestamp_from_header(msg: Any) -> float:
	stamp = getattr(getattr(msg, "header", None), "stamp", None)
	if stamp and stamp != rospy.Time():
		return stamp.to_sec()
	return rospy.Time.now().to_sec()


def _as_vec3(position: Any) -> Tuple[float, float, float]:
	if hasattr(position, "x"):
		return (float(position.x), float(position.y), float(position.z))
	return tuple(position)  # type: ignore[arg-type]


def _as_quat(orientation: Any) -> Tuple[float, float, float, float]:
	if hasattr(orientation, "x"):
		return (
			float(orientation.x),
			float(orientation.y),
			float(orientation.z),
			float(orientation.w),
		)
	return tuple(orientation)  # type: ignore[arg-type]


def _record(timestamp: float, position: Any, orientation: Any) -> PoseRecord:
	return (timestamp, _as_vec3(position), _as_quat(orientation))


def _extract_nav_odometry(msg: Any) -> PoseRecord:
	return _record(_timestamp_from_header(msg), msg.pose.pose.position, msg.pose.pose.orientation)


def _extract_pose_stamped(msg: Any) -> PoseRecord:
	return _record(_timestamp_from_header(msg), msg.pose.position, msg.pose.orientation)


def _extract_pose_with_cov_stamped(msg: Any) -> PoseRecord:
	return _record(_timestamp_from_header(msg), msg.pose.pose.position, msg.pose.pose.orientation)


def _extract_pose_with_cov(msg: Any) -> PoseRecord:
	return _record(_timestamp_from_header(msg), msg.pose.position, msg.pose.orientation)


def _extract_pose_plain(msg: Any) -> PoseRecord:
	return _record(_timestamp_from_header(msg), msg.position, msg.orientation)


def _extract_transform_stamped(msg: Any) -> PoseRecord:
	return _record(_timestamp_from_header(msg), msg.transform.translation, msg.transform.rotation)


MESSAGE_ADAPTERS: Dict[str, PoseExtractor] = {
	"nav_msgs/Odometry": _extract_nav_odometry,
	"geometry_msgs/PoseStamped": _extract_pose_stamped,
	"geometry_msgs/PoseWithCovarianceStamped": _extract_pose_with_cov_stamped,
	"geometry_msgs/PoseWithCovariance": _extract_pose_with_cov,
	"geometry_msgs/Pose": _extract_pose_plain,
	"geometry_msgs/TransformStamped": _extract_transform_stamped,
}


class OdomTumLogger:
	def __init__(self, config: Dict[str, Any]):
		topic = config["topic_name"]
		queue_size = int(config.get("queue_size", 100))
		message_type = config.get("message_type", "nav_msgs/Odometry")

		msg_cls = get_message_class(message_type)
		if msg_cls is None:
			raise ValueError(f"Unable to resolve message type '{message_type}'")

		base_output = Path(os.path.expanduser(config["output_directory"])).resolve()
		dataset_name = config.get("dataset", "default_dataset")
		dataset_slug = re.sub(r"[^A-Za-z0-9]+", "_", dataset_name) or "dataset"
		algorithm_name = config.get("algorithm_name")
		if algorithm_name:
			label_slug = re.sub(r"[^A-Za-z0-9]+", "_", algorithm_name)
		else:
			label_slug = re.sub(r"[^A-Za-z0-9]+", "_", topic.strip("/")) or "odom"
		target_dir = base_output / dataset_slug / label_slug
		target_dir.mkdir(parents=True, exist_ok=True)

		file_prefix = config.get("file_prefix", "odom_export")
		self._file_path = target_dir / f"{file_prefix}.csv"
		self._file = self._file_path.open("w", encoding="utf-8")
		self._received = 0
		self._last_msg_time = rospy.Time.now()
		self._timeout_duration = rospy.Duration(30.0)

		self._message_type = message_type
		self._adapter: Optional[PoseExtractor] = MESSAGE_ADAPTERS.get(message_type)
		if self._adapter is None:
			rospy.logwarn(
				"No dedicated adapter for %s, falling back to generic pose extraction.",
				message_type,
			)
		rospy.loginfo(
			"Writing TUM trajectory to %s (topic=%s, type=%s)",
			self._file_path,
			topic,
			message_type,
		)
		self._sub = rospy.Subscriber(topic, msg_cls, self._odom_callback, queue_size=queue_size)
		self._watchdog = rospy.Timer(rospy.Duration(1.0), self._timer_callback)

	def _generic_extract_pose(self, msg: Any) -> PoseRecord:
		stamp = getattr(getattr(msg, "header", None), "stamp", None)
		if stamp and stamp != rospy.Time():
			timestamp = stamp.to_sec()
		else:
			timestamp = rospy.Time.now().to_sec()

		pose_field = getattr(msg, "pose", None)
		if pose_field is not None and hasattr(pose_field, "pose"):
			pose_field = pose_field.pose

		if pose_field is not None and hasattr(pose_field, "position") and hasattr(pose_field, "orientation"):
			position = pose_field.position
			orientation = pose_field.orientation
			return (
				timestamp,
				(position.x, position.y, position.z),
				(orientation.x, orientation.y, orientation.z, orientation.w),
			)

		transform = getattr(msg, "transform", None)
		if transform is not None:
			translation = getattr(transform, "translation", None)
			rotation = getattr(transform, "rotation", None)
			if translation and rotation:
				return (
					timestamp,
					(translation.x, translation.y, translation.z),
					(rotation.x, rotation.y, rotation.z, rotation.w),
				)

		raise AttributeError(
			"Unsupported message structure for type "
			f"{self._message_type}. Expected a 'pose' or 'transform' attribute."
		)

	def _odom_callback(self, msg: Any) -> None:
		try:
			if self._adapter is not None:
				timestamp, position, orientation = self._adapter(msg)
			else:
				timestamp, position, orientation = self._generic_extract_pose(msg)
		except AttributeError as err:
			rospy.logwarn_throttle(5.0, "Failed to parse %s: %s", self._message_type, err)
			return

		self._last_msg_time = rospy.Time.now()
		line = (
			f"{timestamp:.9f} "
			f"{position[0]:.9f} {position[1]:.9f} {position[2]:.9f} "
			f"{orientation[0]:.9f} {orientation[1]:.9f} {orientation[2]:.9f} {orientation[3]:.9f}\n"
		)

		self._file.write(line)
		self._file.flush()
		self._received += 1
		if self._received % 5 == 0:
			rospy.loginfo("Logged %d poses to %s", self._received, self._file_path.name)

	def _timer_callback(self, _event: rospy.timer.TimerEvent) -> None:
		if rospy.Time.now() - self._last_msg_time > self._timeout_duration:
			rospy.logerr("No messages received for %.1f seconds, shutting down." % self._timeout_duration.to_sec())
			rospy.signal_shutdown("No odometry messages received")

	def close(self) -> None:
		if hasattr(self, "_watchdog"):
			self._watchdog.shutdown()
		if self._file and not self._file.closed:
			rospy.loginfo("Closing TUM trajectory file: %s", self._file_path)
			self._file.close()


def main() -> None:
	config = load_config(CONFIG_PATH)
	rospy.init_node("odom_to_tum_logger", anonymous=False)
	logger = OdomTumLogger(config)
	rospy.on_shutdown(logger.close)
	rospy.loginfo("odom_to_tum_logger is running and waiting for messages...")
	rospy.spin()


if __name__ == "__main__":
	main()
