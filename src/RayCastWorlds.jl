module RayCastWorlds

import RayCaster as RC
import ReinforcementLearningBase as RLBase

abstract type AbstractGame end

function reset! end
function act! end
function cast_rays! end
function get_action_keys end
function get_action_names end
function play! end
function update_top_view! end
function update_camera_view! end

include("utils.jl")
include("collision_detection.jl")
include("rlbase.jl")
include("single_room.jl")

end
