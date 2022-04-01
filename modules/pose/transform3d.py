import torch 
import  numpy as np 


dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor
def  get_point3d(uv, depth, cameraMatrix):
    """_summary_

    Args:
        uv (_type_): _description_
        depth (_type_): _description_
        cameraMatrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    image_height, image_width = depth.shape
    uv_flattened = uv[1].type(dtype_long)*image_width+uv[0].type(dtype_long)
    depth_torch = torch.from_numpy(depth.copy()).type(torch.FloatTensor)
    depth_torch = torch.squeeze(depth_torch, 0)
    depth_torch = depth_torch.view(-1,1)
    
    DEPTH_IM_SCALE = 1000.0 # 
    #depth_vec = torch.index_select(img_a_depth_torch, 0, uv_a_vec_flattened)*1.0/DEPTH_IM_SCALE
    depth_vec = torch.index_select(depth_torch, 0, uv_flattened)*1.0/DEPTH_IM_SCALE
    depth_vec = depth_vec.squeeze(1)
    u_vec = uv[0].type(torch.FloatTensor)*depth_vec
    v_vec = uv[1].type(torch.FloatTensor)*depth_vec
    z_vec = depth_vec
    full_vec = torch.stack((u_vec, v_vec, z_vec))
    K_inv = np.linalg.inv(cameraMatrix)
    K_inv_torch = torch.from_numpy(K_inv).type(torch.FloatTensor)
    point3d = K_inv_torch.mm(full_vec)
    return point3d

def apply_transform_torch(vec3, transform4):
    ones_row = torch.ones_like(vec3[0,:]).type(dtype_float).unsqueeze(0)
    vec4 = torch.cat((vec3,ones_row),0)
    vec4 = transform4.mm(vec4)
    return vec4[0:3]

def invert_transform(transform4):
    transform4_copy = np.copy(transform4)
    R = transform4_copy[0:3,0:3]
    R = np.transpose(R)
    transform4_copy[0:3,0:3] = R
    t = transform4_copy[0:3,3]
    inv_t = -1.0 * np.transpose(R).dot(t)
    transform4_copy[0:3,3] = inv_t
    return transform4_copy