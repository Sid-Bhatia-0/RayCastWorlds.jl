#####
##### unit conversions
#####

wu_to_tu(x_wu) = floor(Int, x_wu) + 1
wu_to_pu(x_wu, wu_per_pu) = fld1(floor(Int, x_wu), wu_per_pu)
pu_to_tu(i_pu, pu_per_tu) = (i_pu - 1) ÷ pu_per_tu + 1

#####
##### navigation
#####

turn_left(direction_au, num_directions) = mod(direction_au + 1, num_directions)
turn_right(direction_au, num_directions) = mod(direction_au - 1, num_directions)

move_forward(position_wu, direction_wu) = position_wu + direction_wu
move_backward(position_wu, direction_wu) = position_wu - direction_wu

#####
##### sampling empty tiles on the tile map
#####

function sample_empty_position(rng, tile_map, region, max_tries)
    position = rand(rng, region)

    for i in 1:max_tries
        if any(@view tile_map[:, position])
            position = rand(rng, region)
        else
            return position
        end
    end

    @warn "Could not sample an empty position in max_tries = $(max_tries). Returning non-empty position: $(position)"

    return position
end

function sample_empty_position(rng, tile_map, region)
    max_tries = 1024 * length(region)
    position = sample_empty_position(rng, tile_map, region, max_tries)
    return position
end

function sample_empty_position(rng, tile_map, max_tries::Integer)
    _, height, width = size(tile_map)
    region = CartesianIndices((1 : height, 1 : width))
    position = sample_empty_position(rng, tile_map, region, max_tries)
    return position
end

function sample_empty_position(rng, tile_map)
    _, height, width = size(tile_map)
    region = CartesianIndices((1 : height, 1 : width))
    max_tries = 1024 * height * width
    position = sample_empty_position(rng, tile_map, region, max_tries)
    return position
end

#####
##### transpose and copy image to minifb frame buffer
#####

function copy_image_to_frame_buffer!(frame_buffer, image)
    height_image, width_image = size(image)
    for j in 1:width_image
        for i in 1:height_image
            frame_buffer[j, i] = image[i, j]
        end
    end

    return nothing
end
