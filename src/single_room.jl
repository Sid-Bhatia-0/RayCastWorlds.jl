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

mutable struct SingleRoomWorld{T, RNG, R}
    tile_map::BitArray{3}
    tile_length::Int
    num_directions::Int
    player_position_wu::CartesianIndex{2}
    player_direction_au::Int
    player_radius_wu::Int
    directions_wu::Vector{CartesianIndex{2}}
    ray_cast_outputs::Vector{Tuple{T, T, Int, Int, Int}}
    goal_position::CartesianIndex{2}
    rng::RNG
    reward::R
    goal_reward::R
    done::Bool
    semi_field_of_view_wu::Rational{Int}
    num_rays::Int
end

function SingleRoomWorld(;
        T = Float32,
        tile_length = 256,
        height_tile_map_tu = 8,
        width_tile_map_tu = 16,
        num_directions = 16,
        player_radius_wu = 32,
        rng = Random.GLOBAL_RNG,
        R = Float32,
        semi_field_of_view_wu = 2//3,
        num_rays = 512,
    )

    @assert iseven(tile_length)

    tile_map = falses(NUM_OBJECTS, height_tile_map_tu, width_tile_map_tu)

    tile_map[WALL, :, 1] .= true
    tile_map[WALL, :, width_tile_map_tu] .= true
    tile_map[WALL, 1, :] .= true
    tile_map[WALL, height_tile_map_tu, :] .= true

    goal_position = CartesianIndex(rand(rng, 2 : height_tile_map_tu - 1), rand(rng, 2 : width_tile_map_tu - 1))
    tile_map[GOAL, goal_position] = true

    directions_wu = Array{CartesianIndex{2}}(undef, num_directions)
    for i in 1:num_directions
        theta_wu = (i - 1) * 2 * pi / num_directions
        directions_wu[i] = CartesianIndex(round(Int, player_radius_wu * cos(theta_wu)), round(Int, player_radius_wu * sin(theta_wu)))
    end

    player_position_tu = RCW.sample_empty_position(rng, tile_map)
    player_position_wu = CartesianIndex(RC.get_tile_end(player_position_tu[1], tile_length) - tile_length ÷ 2, RC.get_tile_end(player_position_tu[2], tile_length) - tile_length ÷ 2)

    player_direction_au = rand(rng, 0 : num_directions - 1)

    ray_cast_outputs = Array{Tuple{T, T, Int, Int, Int}}(undef, num_rays)

    reward = zero(R)
    goal_reward = one(R)
    done = false

    world = SingleRoomWorld(tile_map,
                       tile_length,
                       num_directions,
                       player_position_wu,
                       player_direction_au,
                       player_radius_wu,
                       directions_wu,
                       ray_cast_outputs,
                       goal_position,
                       rng,
                       reward,
                       goal_reward,
                       done,
                       semi_field_of_view_wu,
                       num_rays,
                      )

    RCW.reset!(world)

    return world
end

function RCW.reset!(world::SingleRoomWorld{T}) where {T}
    tile_map = world.tile_map
    tile_length = world.tile_length
    rng = world.rng
    player_radius_wu = world.player_radius_wu
    goal_position = world.goal_position
    num_directions = world.num_directions
    ray_cast_outputs = world.ray_cast_outputs
    directions_wu = world.directions_wu
    semi_field_of_view_wu = world.semi_field_of_view_wu
    _, height_tile_map_tu, width_tile_map_tu = size(tile_map)

    tile_map[GOAL, goal_position] = false

    new_goal_position = CartesianIndex(rand(rng, 2 : height_tile_map_tu - 1), rand(rng, 2 : width_tile_map_tu - 1))
    world.goal_position = new_goal_position
    tile_map[GOAL, new_goal_position] = true

    new_player_position_tu = RCW.sample_empty_position(rng, tile_map)
    new_player_position_wu = CartesianIndex(RC.get_tile_end(new_player_position_tu[1], tile_length) - tile_length ÷ 2, RC.get_tile_end(new_player_position_tu[2], tile_length) - tile_length ÷ 2)
    world.player_position_wu = new_player_position_wu

    new_player_direction_au = rand(rng, 0 : num_directions - 1)
    world.player_direction_au = new_player_direction_au

    world.reward = zero(world.reward)
    world.done = false

    new_player_direction_wu = directions_wu[new_player_direction_au + 1]
    obstacle_tile_map = @view any(tile_map, dims = 1)[1, :, :]
    RC.cast_rays!(ray_cast_outputs, obstacle_tile_map, tile_length, new_player_position_wu[1], new_player_position_wu[2], new_player_direction_wu[1], new_player_direction_wu[2], semi_field_of_view_wu, 1024, RC.FLOAT_DIVISION)

    return nothing
end

function RCW.act!(world::SingleRoomWorld, action)
    @assert action in Base.OneTo(NUM_ACTIONS) "Invalid action: $(action)"

    tile_map = world.tile_map
    tile_length = world.tile_length
    player_direction_au = world.player_direction_au
    player_position_wu = world.player_position_wu
    player_radius_wu = world.player_radius_wu
    num_directions = world.num_directions
    num_rays = length(world.ray_directions_wu)
    goal_map = @view tile_map[GOAL, :, :]

    if action in Base.OneTo(2)
        directions_wu = world.directions_wu
        player_direction_wu = directions_wu[player_direction_au + 1]
        wall_map = @view tile_map[WALL, :, :]

        if action == 1
            new_player_position_wu = RCW.move_forward(player_position_wu, player_direction_wu)
        else
            new_player_position_wu = RCW.move_backward(player_position_wu, player_direction_wu)
        end

        is_colliding_with_goal = RCW.is_player_colliding(goal_map, tile_length, new_player_position_wu, player_radius_wu)
        is_colliding_with_wall = RCW.is_player_colliding(wall_map, tile_length, new_player_position_wu, player_radius_wu)

        if is_colliding_with_goal || is_colliding_with_wall
            if is_colliding_with_goal
                world.reward = world.goal_reward
                world.done = true
            else
                world.reward = zero(world.reward)
                world.done = false
            end
        else
            world.player_position_wu = new_player_position_wu
            world.reward = zero(world.reward)
            world.done = false
        end
    else
        if action == 3
            new_player_direction_au = RCW.turn_left(player_direction_au, num_directions)
        else
            new_player_direction_au = RCW.turn_right(player_direction_au, num_directions)
        end

        world.player_direction_au = new_player_direction_au
        world.reward = zero(world.reward)
        world.done = false
    end

    return nothing
end

#####
##### drawing & playing
#####

const NUM_VIEWS = 2
const CAMERA_VIEW = 1
const TOP_VIEW = 2

struct SingleRoom{T, RNG, R, C} <: RCW.AbstractGame
    world::SingleRoomWorld{T, RNG, R}
    top_view::Array{C, 2}
    camera_view::Array{C, 2}
    tile_map_colors::NTuple{NUM_OBJECTS + 1, C}
    ray_color::C
    player_color::C
    floor_color::C
    ceiling_color::C
    wall_dim_1_color::C
    wall_dim_2_color::C
    goal_dim_1_color::C
    goal_dim_2_color::C
    camera_height_tile_wu::T
    height_camera_view_pu::Int
end

function SingleRoom(;
        T = Float32,
        tile_length = 256,
        height_tile_map_tu = 8,
        width_tile_map_tu = 16,
        num_directions = 16,
        player_radius_wu = 32,
        rng = Random.GLOBAL_RNG,
        R = Float32,
        semi_field_of_view_wu = 2//3,
        num_rays = 512,
        pu_per_tu = 32,
        camera_height_tile_wu = convert(T, 1),
        height_camera_view_pu = 256,
    )

    C = UInt32

    world = SingleRoomWorld(T = T,
                           tile_length = tile_length,
                           height_tile_map_tu = height_tile_map_tu,
                           width_tile_map_tu = width_tile_map_tu,
                           num_directions = num_directions,
                           player_radius_wu = player_radius_wu,
                           rng = rng,
                           R = R,
                           semi_field_of_view_wu = semi_field_of_view_wu,
                           num_rays = num_rays,
                          )

    tile_map_colors = (0x00FFFFFF, 0x00FF0000, 0x00000000)
    ray_color = 0x00808080
    player_color = 0x00c0c0c0
    floor_color = 0x00404040
    ceiling_color = 0x00FFFFFF
    wall_dim_1_color = 0x00808080
    wall_dim_2_color = 0x00c0c0c0
    goal_dim_1_color = 0x00800000
    goal_dim_2_color = 0x00c00000

    RCW.cast_rays!(world)

    camera_view = Array{C}(undef, height_camera_view_pu, num_rays)

    top_view = Array{C}(undef, height_tile_map_tu * pu_per_tu, width_tile_map_tu * pu_per_tu)

    env = SingleRoom(world,
                          top_view,
                          camera_view,
                          tile_map_colors,
                          ray_color,
                          player_color,
                          floor_color,
                          ceiling_color,
                          wall_dim_1_color,
                          wall_dim_2_color,
                          goal_dim_1_color,
                          goal_dim_2_color,
                          camera_height_tile_wu,
                          height_camera_view_pu,
                         )

    RCW.update_camera_view!(env)
    RCW.update_top_view!(env)

    return env
end

function RCW.reset!(env::SingleRoom)
    RCW.reset!(env.world)
    RCW.update_top_view!(env)
    RCW.update_camera_view!(env)
    return nothing
end

function RCW.act!(env::SingleRoom, action)
    world = env.world
    RCW.act!(world, action)
    RCW.cast_rays!(world)
    RCW.update_top_view!(env)
    RCW.update_camera_view!(env)
    return nothing
end

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
                color = colors[end]
            else
                color = colors[object]
            end

            SD.draw!(top_view, shape, color)

            top_view[i_top_left, j_top_left : j_top_left + pu_per_tu - 1] .= 0x00cccccc
            top_view[i_top_left + pu_per_tu - 1, j_top_left : j_top_left + pu_per_tu - 1] .= 0x00cccccc
            top_view[i_top_left : i_top_left + pu_per_tu - 1, j_top_left] .= 0x00cccccc
            top_view[i_top_left : i_top_left + pu_per_tu - 1, j_top_left + pu_per_tu - 1] .= 0x00cccccc
        end
    end

    return nothing
end

function get_normalized_dot_product(v1, v2)
    x1 = v1[1]
    y1 = v1[2]
    x2 = v2[1]
    y2 = v2[2]
    return (x1 * x2 + y1 * y2) / (hypot(x1, y1) * hypot(x2, y2))
end

function RCW.update_camera_view!(env::SingleRoom)
    world = env.world
    camera_view = env.camera_view
    floor_color = env.floor_color
    ceiling_color = env.ceiling_color
    wall_dim_1_color = env.wall_dim_1_color
    wall_dim_2_color = env.wall_dim_2_color
    goal_dim_1_color = env.goal_dim_1_color
    goal_dim_2_color = env.goal_dim_2_color
    camera_height_tile_wu = env.camera_height_tile_wu

    tile_map = world.tile_map
    tile_length = world.tile_length
    player_direction_au = world.player_direction_au
    player_position_wu = world.player_position_wu
    ray_directions_wu = world.ray_directions_wu
    num_rays = length(ray_directions_wu)
    num_directions = world.num_directions
    ray_stop_position_tu = world.ray_stop_position_tu
    ray_hit_dimension = world.ray_hit_dimension
    ray_distance_wu = world.ray_distance_wu
    directions_wu = world.directions_wu
    semi_field_of_view_wu = world.semi_field_of_view_wu

    _, height_tile_map_tu, width_tile_map_tu = size(tile_map)
    height_camera_view_pu, width_camera_view_pu = size(camera_view)

    player_direction_wu = directions_wu[player_direction_au + 1]
    for i in 1:num_rays
        ray_direction_wu = ray_directions_wu[i]

        normalized_projected_distance_wu = ray_distance_wu[i] * get_normalized_dot_product(player_direction_wu, ray_direction_wu)

        height_line = camera_height_tile_wu * num_rays / (2 * semi_field_of_view_wu * normalized_projected_distance_wu)

        if isfinite(height_line)
            height_line_pu = floor(Int, height_line)
        else
            height_line_pu = height_camera_view_pu
        end

        hit_dimension = ray_hit_dimension[i]
        ray_stop_position_i_tu = ray_stop_position_tu[1, i]
        ray_stop_position_j_tu = ray_stop_position_tu[2, i]

        if tile_map[WALL, ray_stop_position_i_tu, ray_stop_position_j_tu]
            if hit_dimension == 1
                color = wall_dim_1_color
            else hit_dimension == 2
                color = wall_dim_2_color
            end
        else
            if hit_dimension == 1
                color = goal_dim_1_color
            else hit_dimension == 2
                color = goal_dim_2_color
            end
        end

        if height_line_pu >= height_camera_view_pu - 1
            camera_view[:, i] .= color
        else
            padding_pu = (height_camera_view_pu - height_line_pu) ÷ 2
            camera_view[1:padding_pu, i] .= ceiling_color
            camera_view[padding_pu + 1 : end - padding_pu, i] .= color
            camera_view[end - padding_pu + 1 : end, i] .= floor_color
        end
    end

    return nothing
end

function RCW.update_top_view!(env::SingleRoom)
    world = env.world
    top_view = env.top_view
    tile_map_colors = env.tile_map_colors
    ray_color = env.ray_color
    player_color = env.player_color
    tile_map = world.tile_map
    tile_length = world.tile_length
    player_direction_au = world.player_direction_au
    player_position_wu = world.player_position_wu
    ray_directions_wu = world.ray_directions_wu
    num_rays = length(ray_directions_wu)
    num_directions = world.num_directions
    player_radius_wu = world.player_radius_wu
    ray_stop_position_tu = world.ray_stop_position_tu
    ray_hit_dimension = world.ray_hit_dimension
    ray_distance_wu = world.ray_distance_wu
    directions_wu = world.directions_wu

    _, height_tile_map_tu, width_tile_map_tu = size(tile_map)
    height_top_view_pu, width_top_view_pu = size(top_view)

    pu_per_tu = height_top_view_pu ÷ height_tile_map_tu

    wu_per_pu = tile_length ÷ pu_per_tu
    i_player_position_pu = RCW.wu_to_pu(player_position_wu[1], wu_per_pu)
    j_player_position_pu = RCW.wu_to_pu(player_position_wu[2], wu_per_pu)
    player_radius_pu = RCW.wu_to_pu(player_radius_wu, wu_per_pu)

    draw_tile_map!(top_view, tile_map, tile_map_colors)

    for i in 1:num_rays
        ray_direction_wu = ray_directions_wu[i]
        ray_direction_wu_norm = hypot(ray_direction_wu[1], ray_direction_wu[2])
        x_ray_direction_wu_normalized = ray_direction_wu[1] / ray_direction_wu_norm
        y_ray_direction_wu_normalized = ray_direction_wu[2] / ray_direction_wu_norm
        i_ray_stop_pu = RCW.wu_to_pu(player_position_wu[1] + ray_distance_wu[i] * x_ray_direction_wu_normalized, wu_per_pu)
        j_ray_stop_pu = RCW.wu_to_pu(player_position_wu[2] + ray_distance_wu[i] * y_ray_direction_wu_normalized, wu_per_pu)
        SD.draw!(top_view, SD.Line(SD.Point(round(Int, i_player_position_pu), round(Int, j_player_position_pu)), SD.Point(round(Int, i_ray_stop_pu), round(Int, j_ray_stop_pu))), ray_color)
    end

    SD.draw!(top_view, SD.Circle(SD.Point(i_player_position_pu - player_radius_pu, j_player_position_pu - player_radius_pu), 2 * player_radius_pu), player_color)

    return nothing
end

RCW.get_action_keys(env::SingleRoom) = (MFB.KB_KEY_W, MFB.KB_KEY_S, MFB.KB_KEY_A, MFB.KB_KEY_D)
RCW.get_action_names(env::SingleRoom) = (:MOVE_FORWARD, :MOVE_BACKWARD, :TURN_LEFT, :TURN_RIGHT)

function RCW.play!(game::SingleRoom)
    world = game.world
    top_view = game.top_view
    camera_view = game.camera_view
    tile_map_colors = game.tile_map_colors
    ray_color = game.ray_color
    player_color = game.player_color
    floor_color = game.floor_color
    ceiling_color = game.ceiling_color
    wall_dim_1_color = game.wall_dim_1_color
    wall_dim_2_color = game.wall_dim_2_color

    height_top_view_pu, width_top_view_pu = size(top_view)
    height_camera_view_pu, width_camera_view_pu = size(camera_view)

    height_image = max(height_top_view_pu, height_camera_view_pu)
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
            @show world.reward
            @show world.done
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

RLBase.reward(env::RCW.RLBaseEnv{E}) where {E <: SingleRoom} = env.env.world.reward
RLBase.is_terminated(env::RCW.RLBaseEnv{E}) where {E <: SingleRoom} = env.env.world.done

end # module
