extends Skeleton3D

@export var udp_bind_address: String = "0.0.0.0"
@export var udp_port: int = 7000
@export var pose_blend: float = 0.30
@export var min_confidence: float = 0.08
@export var print_missing_bones_once: bool = true

var _server := UDPServer.new()
var _peer: PacketPeerUDP
var _bone_index_by_name := {}
var _missing_once := {}


func _ready() -> void:
	for i in range(get_bone_count()):
		_bone_index_by_name[get_bone_name(i)] = i

	var err := _server.listen(udp_port, udp_bind_address)
	if err != OK:
		push_error("AvatarPoseReceiver listen failed on %s:%d (err=%d)" % [udp_bind_address, udp_port, err])
		return

	print("AvatarPoseReceiver listening on %s:%d" % [udp_bind_address, udp_port])


func _process(_delta: float) -> void:
	_server.poll()
	if _server.is_connection_available():
		_peer = _server.take_connection()

	if _peer == null:
		return

	while _peer.get_available_packet_count() > 0:
		var packet := _peer.get_packet()
		var text := packet.get_string_from_utf8()
		var json := JSON.new()
		var parse_err := json.parse(text)
		if parse_err != OK:
			continue

		var data: Variant = json.data
		if typeof(data) != TYPE_DICTIONARY:
			continue
		var bones: Variant = data.get("bones", {})
		if typeof(bones) != TYPE_DICTIONARY:
			continue

		_apply_bones(bones)


func _apply_bones(bones: Dictionary) -> void:
	for bone_name in bones.keys():
		var bone_payload: Variant = bones[bone_name]
		if typeof(bone_payload) != TYPE_DICTIONARY:
			continue

		var conf := float(bone_payload.get("confidence", 1.0))
		if conf < min_confidence:
			continue

		if not _bone_index_by_name.has(bone_name):
			if print_missing_bones_once and not _missing_once.has(bone_name):
				_missing_once[bone_name] = true
				print("AvatarPoseReceiver: bone not found in Skeleton3D -> ", bone_name)
			continue

		var q_arr: Variant = bone_payload.get("rotation_quat_wxyz", [])
		if typeof(q_arr) != TYPE_ARRAY:
			continue
		if q_arr.size() != 4:
			continue

		var w := float(q_arr[0])
		var x := float(q_arr[1])
		var y := float(q_arr[2])
		var z := float(q_arr[3])
		var target_q := Quaternion(x, y, z, w).normalized()

		var bone_idx := int(_bone_index_by_name[bone_name])
		var current_q := get_bone_pose_rotation(bone_idx)
		var blended_q := current_q.slerp(target_q, clampf(pose_blend, 0.0, 1.0)).normalized()
		set_bone_pose_rotation(bone_idx, blended_q)
