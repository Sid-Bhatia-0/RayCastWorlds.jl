struct StdSquare{T}
    half_side::T
end

struct StdCircle{T}
    radius::T
end

function get_projection(square::StdSquare, position)
    half_side = square.half_side
    return typeof(position)(clamp(position[1], -half_side, half_side), clamp(position[2], -half_side, half_side))
end

function is_colliding(square::StdSquare, circle::StdCircle, position)
    radius = circle.radius
    projection = get_projection(square, position)
    vec = typeof(position)(position[1] - projection[1], position[2] - projection[2])
    return vec[1] * vec[1] + vec[2] * vec[2] < radius * radius
end

function is_player_colliding(obstacle_tile_map, tile_length, player_position_wu, player_radius_wu)
    height_tile_map_tu, width_tile_map_tu = size(obstacle_tile_map)

    square = StdSquare(tile_length ÷ 2)
    circle = StdCircle(player_radius_wu)

    i_player_position_tu_min = fld1(player_position_wu[1] - player_radius_wu, tile_length)
    j_player_position_tu_min = fld1(player_position_wu[2] - player_radius_wu, tile_length)

    i_player_position_tu_max = fld1(player_position_wu[1] + player_radius_wu, tile_length)
    j_player_position_tu_max = fld1(player_position_wu[2] + player_radius_wu, tile_length)

    for j in j_player_position_tu_min : j_player_position_tu_max
        for i in i_player_position_tu_min : i_player_position_tu_max
            tile_position_wu = typeof(player_position_wu)(RC.get_tile_end(i, tile_length) - tile_length ÷ 2, RC.get_tile_end(j, tile_length) - tile_length ÷ 2)
            if obstacle_tile_map[i, j] && is_colliding(square, circle, player_position_wu - tile_position_wu)
                return true
            end
        end
    end

    return false
end
