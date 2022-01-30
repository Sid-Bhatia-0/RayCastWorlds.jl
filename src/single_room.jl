module SingleRoomModule

import MiniFB as MFB
import Random
import ..RayCastWorlds as RCW
import RayCaster as RC
import ReinforcementLearningBase as RLBase
import SimpleDraw as SD

#####
##### game logic
#####

const NUM_OBJECTS = 2
const WALL = 1
const GOAL = 2
const NUM_ACTIONS = 4

const NUM_VIEWS = 2
const CAMERA_VIEW = 1
const TOP_VIEW = 2

function get_player_direction(player_angle, num_angles, player_radius)
    theta_wu = player_angle * 2 * pi / num_angles
    return CartesianIndex(round(Int, player_radius * cos(theta_wu)), round(Int, player_radius * sin(theta_wu)))
end

mutable struct SingleRoom{T, RNG, R, C} <: RCW.AbstractGame
    tile_map::BitArray{3}
    tile_length::Int
    num_angles::Int
    player_position::CartesianIndex{2}
    player_angle::Int
    player_radius::Int
    ray_cast_outputs::Vector{Tuple{T, T, Int, Int, Int, Int, Int}}
    goal_tile::CartesianIndex{2}
    rng::RNG
    reward::R
    goal_reward::R
    done::Bool
    semi_field_of_view_ratio::Rational{Int}
    num_rays::Int

    top_view::Array{C, 2}
    camera_view::Array{C, 2}
    top_view_colors::NamedTuple{(:wall, :goal, :empty, :ray, :player, :border), NTuple{NUM_OBJECTS + 4, C}}
    camera_view_colors::NamedTuple{(:wall1, :wall2, :goal1, :goal2, :floor, :ceiling), NTuple{2 * NUM_OBJECTS + 2, C}}
    tile_aspect_ratio_camera_view::Rational{Int}
    height_camera_view::Int
end

function SingleRoom(;
        T = Float32,
        tile_length = 256,
        height_tile_map_tu = 8,
        width_tile_map_tu = 16,
        num_angles = 64,
        player_radius = 32,
        rng = Random.GLOBAL_RNG,
        R = Float32,
        semi_field_of_view_ratio = 2//3,
        num_rays = 512,
        pu_per_tu = 32,
        tile_aspect_ratio_camera_view = 1//1,
        height_camera_view = 256,
        top_view_colors = (wall = 0x00FFFFFF, goal = 0x00FF0000, empty = 0x00000000, ray = 0x00808080, player = 0x00c0c0c0, border = 0x00cccccc),
        camera_view_colors = (wall1 = 0x00808080, wall2 = 0x00c0c0c0, goal1 = 0x00800000, goal2 = 0x00c00000, floor = 0x00404040, ceiling = 0x00FFFFFF),
    )

    @assert iseven(tile_length)

    tile_map = falses(NUM_OBJECTS, height_tile_map_tu, width_tile_map_tu)

    tile_map[WALL, :, 1] .= true
    tile_map[WALL, :, width_tile_map_tu] .= true
    tile_map[WALL, 1, :] .= true
    tile_map[WALL, height_tile_map_tu, :] .= true

    goal_tile = CartesianIndex(rand(rng, 2 : height_tile_map_tu - 1), rand(rng, 2 : width_tile_map_tu - 1))
    tile_map[GOAL, goal_tile] = true

    player_tile = RCW.sample_empty_position(rng, tile_map)
    player_position = CartesianIndex(RC.get_tile_end(player_tile[1], tile_length) - tile_length ÷ 2, RC.get_tile_end(player_tile[2], tile_length) - tile_length ÷ 2)

    player_angle = rand(rng, 0 : num_angles - 1)

    ray_cast_outputs = Array{Tuple{T, T, Int, Int, Int, Int, Int}}(undef, num_rays)

    reward = zero(R)
    goal_reward = one(R)
    done = false

    C = UInt32

    camera_view = Array{C}(undef, height_camera_view, num_rays)

    top_view = Array{C}(undef, height_tile_map_tu * pu_per_tu, width_tile_map_tu * pu_per_tu)

    env = SingleRoom(
                        tile_map,
                        tile_length,
                        num_angles,
                        player_position,
                        player_angle,
                        player_radius,
                        ray_cast_outputs,
                        goal_tile,
                        rng,
                        reward,
                        goal_reward,
                        done,
                        semi_field_of_view_ratio,
                        num_rays,

                        top_view,
                        camera_view,
                        top_view_colors,
                        camera_view_colors,
                        tile_aspect_ratio_camera_view,
                        height_camera_view,
                        )

    RCW.reset!(env)
    RCW.update_camera_view!(env)
    RCW.update_top_view!(env)

    return env
end

function RCW.reset!(env::SingleRoom{T}) where {T}
    tile_map = env.tile_map
    tile_length = env.tile_length
    rng = env.rng
    player_radius = env.player_radius
    goal_tile = env.goal_tile
    num_angles = env.num_angles
    ray_cast_outputs = env.ray_cast_outputs
    semi_field_of_view_ratio = env.semi_field_of_view_ratio
    _, height_tile_map_tu, width_tile_map_tu = size(tile_map)

    tile_map[GOAL, goal_tile] = false

    new_goal_tile = CartesianIndex(rand(rng, 2 : height_tile_map_tu - 1), rand(rng, 2 : width_tile_map_tu - 1))
    env.goal_tile = new_goal_tile
    tile_map[GOAL, new_goal_tile] = true

    new_player_tile = RCW.sample_empty_position(rng, tile_map)
    new_player_position = CartesianIndex(RC.get_tile_end(new_player_tile[1], tile_length) - tile_length ÷ 2, RC.get_tile_end(new_player_tile[2], tile_length) - tile_length ÷ 2)
    env.player_position = new_player_position

    new_player_angle = rand(rng, 0 : num_angles - 1)
    env.player_angle = new_player_angle

    env.reward = zero(env.reward)
    env.done = false

    new_player_direction_wu = get_player_direction(new_player_angle, num_angles, player_radius)
    obstacle_tile_map = @view any(tile_map, dims = 1)[1, :, :]
    RC.cast_rays!(ray_cast_outputs, obstacle_tile_map, tile_length, new_player_position[1], new_player_position[2], new_player_direction_wu[1], new_player_direction_wu[2], semi_field_of_view_ratio, 1024, RC.FLOAT_DIVISION)

    RCW.update_top_view!(env)
    RCW.update_camera_view!(env)

    return nothing
end

function RCW.act!(env::SingleRoom, action)
    @assert action in Base.OneTo(NUM_ACTIONS) "Invalid action: $(action)"

    tile_map = env.tile_map
    tile_length = env.tile_length
    player_angle = env.player_angle
    player_position = env.player_position
    player_radius = env.player_radius
    num_angles = env.num_angles
    num_rays = env.num_rays
    goal_map = @view tile_map[GOAL, :, :]

    num_angles = env.num_angles
    ray_cast_outputs = env.ray_cast_outputs
    semi_field_of_view_ratio = env.semi_field_of_view_ratio

    if action in Base.OneTo(2)
        player_direction_wu = get_player_direction(player_angle, num_angles, player_radius)
        wall_map = @view tile_map[WALL, :, :]

        if action == 1
            new_player_position = RCW.move_forward(player_position, player_direction_wu)
        else
            new_player_position = RCW.move_backward(player_position, player_direction_wu)
        end

        is_colliding_with_goal = RCW.is_player_colliding(goal_map, tile_length, new_player_position, player_radius)
        is_colliding_with_wall = RCW.is_player_colliding(wall_map, tile_length, new_player_position, player_radius)

        if is_colliding_with_goal || is_colliding_with_wall
            if is_colliding_with_goal
                env.reward = env.goal_reward
                env.done = true
            else
                env.reward = zero(env.reward)
                env.done = false
            end
        else
            env.player_position = new_player_position
            env.reward = zero(env.reward)
            env.done = false
        end
    else
        if action == 3
            new_player_angle = RCW.turn_left(player_angle, num_angles)
        else
            new_player_angle = RCW.turn_right(player_angle, num_angles)
        end

        env.player_angle = new_player_angle
        env.reward = zero(env.reward)
        env.done = false
    end

    x_ray_start = env.player_position[1]
    y_ray_start = env.player_position[2]

    player_direction_wu = get_player_direction(env.player_angle, num_angles, player_radius)
    x_ray_direction = player_direction_wu[1]
    y_ray_direction = player_direction_wu[2]

    obstacle_tile_map = @view any(tile_map, dims = 1)[1, :, :]
    RC.cast_rays!(ray_cast_outputs, obstacle_tile_map, tile_length, x_ray_start, y_ray_start, x_ray_direction, y_ray_direction, semi_field_of_view_ratio, 1024, RC.FLOAT_DIVISION)
    RCW.update_top_view!(env)
    RCW.update_camera_view!(env)

    return nothing
end

#####
##### drawing & playing
#####

function draw_tile_map!(top_view, tile_map, colors)
    _, height_tile_map_tu, width_tile_map_tu = size(tile_map)
    height_top_view_pu, width_top_view_pu = size(top_view)

    pu_per_tu = height_top_view_pu ÷ height_tile_map_tu

    for j in 1:width_tile_map_tu
        for i in 1:height_tile_map_tu
            i_top_left = (i - 1) * pu_per_tu + 1
            j_top_left = (j - 1) * pu_per_tu + 1

            shape = SD.FilledRectangle(SD.Point(i_top_left, j_top_left), pu_per_tu, pu_per_tu)

            object = findfirst(@view tile_map[:, i, j])
            if isnothing(object)
                color = colors[:empty]
            else
                color = colors[object]
            end

            SD.draw!(top_view, shape, color)

            border_color = colors[:border]
            top_view[i_top_left, j_top_left : j_top_left + pu_per_tu - 1] .= border_color
            top_view[i_top_left + pu_per_tu - 1, j_top_left : j_top_left + pu_per_tu - 1] .= border_color
            top_view[i_top_left : i_top_left + pu_per_tu - 1, j_top_left] .= border_color
            top_view[i_top_left : i_top_left + pu_per_tu - 1, j_top_left + pu_per_tu - 1] .= border_color
        end
    end

    return nothing
end

get_normalized_dot_product(x1, y1, x2, y2) = (x1 * x2 + y1 * y2) / (hypot(x1, y1) * hypot(x2, y2))

function RCW.update_camera_view!(env::SingleRoom)
    camera_view = env.camera_view
    tile_aspect_ratio_camera_view = env.tile_aspect_ratio_camera_view

    tile_map = env.tile_map
    tile_length = env.tile_length
    player_angle = env.player_angle
    player_position = env.player_position
    player_radius = env.player_radius
    num_rays = env.num_rays
    num_angles = env.num_angles
    semi_field_of_view_ratio = env.semi_field_of_view_ratio
    ray_cast_outputs = env.ray_cast_outputs
    camera_view_colors = env.camera_view_colors

    _, height_tile_map_tu, width_tile_map_tu = size(tile_map)
    height_camera_view, width_camera_view_pu = size(camera_view)

    player_direction_wu = get_player_direction(player_angle, num_angles, player_radius)
    for i in 1:num_rays
        x_ray_stop, y_ray_stop, i_ray_hit_tile, j_ray_hit_tile, hit_dimension, x_ray_direction, y_ray_direction = ray_cast_outputs[i]

        ray_distance_wu = hypot(x_ray_stop - player_position[1], y_ray_stop - player_position[2])
        normalized_projected_distance_wu = ray_distance_wu * get_normalized_dot_product(player_direction_wu[1], player_direction_wu[2], x_ray_direction, y_ray_direction)

        height_line = tile_aspect_ratio_camera_view * tile_length * num_rays / (2 * semi_field_of_view_ratio * normalized_projected_distance_wu)

        if isfinite(height_line)
            height_line_pu = floor(Int, height_line)
        else
            height_line_pu = height_camera_view
        end

        object = findfirst(@view tile_map[:, i_ray_hit_tile, j_ray_hit_tile])
        if isnothing(object)
            color = camera_view_colors[:ceiling]
        else
            color = camera_view_colors[2 * (object - 1) + hit_dimension]
        end

        if height_line_pu >= height_camera_view - 1
            camera_view[:, i] .= color
        else
            padding_pu = (height_camera_view - height_line_pu) ÷ 2
            camera_view[1:padding_pu, i] .= camera_view_colors[:ceiling]
            camera_view[padding_pu + 1 : end - padding_pu, i] .= color
            camera_view[end - padding_pu + 1 : end, i] .= camera_view_colors[:floor]
        end
    end

    return nothing
end

function RCW.update_top_view!(env::SingleRoom)
    top_view = env.top_view
    top_view_colors = env.top_view_colors

    tile_map = env.tile_map
    tile_length = env.tile_length
    player_angle = env.player_angle
    player_position = env.player_position
    num_rays = env.num_rays
    num_angles = env.num_angles
    player_radius = env.player_radius
    ray_cast_outputs = env.ray_cast_outputs

    _, height_tile_map_tu, width_tile_map_tu = size(tile_map)
    height_top_view_pu, width_top_view_pu = size(top_view)

    pu_per_tu = height_top_view_pu ÷ height_tile_map_tu

    wu_per_pu = tile_length ÷ pu_per_tu
    i_player_position_pu = RCW.wu_to_pu(player_position[1], wu_per_pu)
    j_player_position_pu = RCW.wu_to_pu(player_position[2], wu_per_pu)
    player_radius_pixel_units = RCW.wu_to_pu(player_radius, wu_per_pu)

    draw_tile_map!(top_view, tile_map, top_view_colors)

    for i in 1:num_rays
        x_ray_stop, y_ray_stop, i_ray_hit_tile, j_ray_hit_tile, hit_dimension, _, _ = ray_cast_outputs[i]
        i_ray_stop_pu = RCW.wu_to_pu(x_ray_stop, wu_per_pu)
        j_ray_stop_pu = RCW.wu_to_pu(y_ray_stop, wu_per_pu)

        SD.draw!(top_view, SD.Line(SD.Point(i_player_position_pu, j_player_position_pu), SD.Point(i_ray_stop_pu, j_ray_stop_pu)), top_view_colors[:ray])
    end

    SD.draw!(top_view, SD.Circle(SD.Point(i_player_position_pu - player_radius_pixel_units, j_player_position_pu - player_radius_pixel_units), 2 * player_radius_pixel_units), top_view_colors[:player])

    return nothing
end

RCW.get_action_keys(env::SingleRoom) = (MFB.KB_KEY_W, MFB.KB_KEY_S, MFB.KB_KEY_A, MFB.KB_KEY_D)
RCW.get_action_names(env::SingleRoom) = (:MOVE_FORWARD, :MOVE_BACKWARD, :TURN_LEFT, :TURN_RIGHT)

function RCW.play!(game::SingleRoom)
    top_view = game.top_view
    camera_view = game.camera_view
    top_view_colors = game.top_view_colors
    camera_view_colors = game.camera_view_colors

    height_top_view_pu, width_top_view_pu = size(top_view)
    height_camera_view, width_camera_view_pu = size(camera_view)

    height_image = max(height_top_view_pu, height_camera_view)
    width_image = max(width_top_view_pu, width_camera_view_pu)

    frame_buffer = zeros(UInt32, width_image, height_image)

    window = MFB.mfb_open(String(nameof(typeof(game))), width_image, height_image)

    action_keys = RCW.get_action_keys(game)

    current_view = CAMERA_VIEW
    if current_view == CAMERA_VIEW
        RCW.copy_image_to_frame_buffer!(frame_buffer, camera_view)
    elseif current_view == TOP_VIEW
        RCW.copy_image_to_frame_buffer!(frame_buffer, top_view)
    end

    steps_taken = 0

    function keyboard_callback(window, key, mod, is_pressed)::Cvoid
        if is_pressed
            println("*******************************")
            @show key

            if key == MFB.KB_KEY_Q
                MFB.mfb_close(window)
                return nothing
            elseif key == MFB.KB_KEY_R
                RCW.reset!(game)
                steps_taken = 0
            elseif key == MFB.KB_KEY_V
                current_view = mod1(current_view + 1, NUM_VIEWS)
                fill!(frame_buffer, 0x00000000)
            elseif key in action_keys
                action = findfirst(==(key), action_keys)
                RCW.act!(game, action)
                steps_taken += 1
            else
                @warn "No keybinding exists for $(key)"
            end

            if current_view == CAMERA_VIEW
                RCW.copy_image_to_frame_buffer!(frame_buffer, camera_view)
            elseif current_view == TOP_VIEW
                RCW.copy_image_to_frame_buffer!(frame_buffer, top_view)
            end

            @show steps_taken
            @show game.reward
            @show game.done
        end

        return nothing
    end

    MFB.mfb_set_keyboard_callback(window, keyboard_callback)

    while MFB.mfb_wait_sync(window)
        state = MFB.mfb_update(window, frame_buffer)

        if state != MFB.STATE_OK
            break;
        end
    end

    return nothing
end

#####
##### RLBase API
#####

RLBase.StateStyle(env::RCW.RLBaseEnv{E}) where {E <: SingleRoom} = RLBase.Observation{Any}()
RLBase.state_space(env::RCW.RLBaseEnv{E}, ::RLBase.Observation) where {E <: SingleRoom} = nothing
RLBase.state(env::RCW.RLBaseEnv{E}, ::RLBase.Observation) where {E <: SingleRoom} = env.env.camera_view

RLBase.reset!(env::RCW.RLBaseEnv{E}) where {E <: SingleRoom} = RCW.reset!(env.env)

RLBase.action_space(env::RCW.RLBaseEnv{E}) where {E <: SingleRoom} = Base.OneTo(NUM_ACTIONS)
(env::RCW.RLBaseEnv{E})(action) where {E <: SingleRoom} = RCW.act!(env.env, action)

RLBase.reward(env::RCW.RLBaseEnv{E}) where {E <: SingleRoom} = env.env.reward
RLBase.is_terminated(env::RCW.RLBaseEnv{E}) where {E <: SingleRoom} = env.env.done

end # module
