import os
import configargparse
import smplx
import numpy as np 
import torch
from utils.fast_render import generate_silent_videos
from utils.media import add_audio_to_video

def render_one_sequence_with_face(
         npz_path,
         output_dir,
         audio_path,
         model_folder="./pretrained/smplx_models/",
         model_type='smplx',
         gender='NEUTRAL_2020',
         ext='npz',
         num_betas=300,
         num_expression_coeffs=100,
         use_face_contour=False,
         args=None):    
    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext, use_pca=False).cuda()
    
    data_np_body = np.load(npz_path, allow_pickle=True)
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    faces = np.load(f"{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]
    seconds = 1
    n = data_np_body["poses"].shape[0]
    beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).cuda()
    beta = beta.repeat(n, 1)
    expression = torch.from_numpy(data_np_body["expressions"][:n]).to(torch.float32).cuda()
    jaw_pose = torch.from_numpy(data_np_body["poses"][:n, 66:69]).to(torch.float32).cuda()
    pose = torch.from_numpy(data_np_body["poses"][:n]).to(torch.float32).cuda()
    transl_np = data_np_body["trans"][:n]
    transl_np[:, :] = transl_np[0, :]
    transl = torch.from_numpy(transl_np).to(torch.float32).cuda()
    output = model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose,
        global_orient=pose[:,:3], body_pose=pose[:,3:21*3+3], left_hand_pose=pose[:,25*3:40*3], right_hand_pose=pose[:,40*3:55*3],
        leye_pose=pose[:, 69:72], 
        reye_pose=pose[:, 72:75],
        return_verts=True)
    vertices_all = output["vertices"].cpu().detach().numpy()

    beta1 = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).cuda()
    beta1 = beta1.repeat(n, 1)
    expression1 = torch.from_numpy(data_np_body["expressions"][:n]).to(torch.float32).cuda()
    zero_pose = np.zeros_like(data_np_body["poses"])
    jaw_pose1 = torch.from_numpy(zero_pose[:n,66:69]).to(torch.float32).cuda()
    pose1 = torch.from_numpy(zero_pose[:n]).to(torch.float32).cuda()
    zero_trans = np.zeros_like(data_np_body["trans"])
    transl1 = torch.from_numpy(zero_trans[:n]).to(torch.float32).cuda()
    output1 = model(betas=beta1, transl=transl1, expression=expression1, jaw_pose=jaw_pose1, 
        global_orient=pose1[:,:3], body_pose=pose1[:,3:21*3+3], left_hand_pose=pose1[:,25*3:40*3], right_hand_pose=pose1[:,40*3:55*3],      
        leye_pose=pose1[:, 69:72], 
        reye_pose=pose1[:, 72:75],
        return_verts=True)
    vertices1_all = output1["vertices"].cpu().detach().numpy()*8
    trans_down = np.zeros_like(vertices1_all)
    trans_down[:, :, 1] = 1.55
    vertices1_all = vertices1_all - trans_down

    seconds = vertices_all.shape[0]//30
    silent_video_file_path = generate_silent_videos(args.render_video_fps,
                                                                args.render_video_width,
                                                                args.render_video_height,
                                                                args.render_concurrent_num,
                                                                args.render_tmp_img_filetype,
                                                                int(seconds*args.render_video_fps), 
                                                                vertices1_all,
                                                                vertices_all,
                                                                faces,
                                                                output_dir)
    base_filename_without_ext = os.path.splitext(os.path.basename(npz_path))[0]
    final_clip = os.path.join(output_dir, f"{base_filename_without_ext}.mp4")
    if audio_path:
        add_audio_to_video(silent_video_file_path, audio_path, final_clip)
        os.remove(silent_video_file_path)
        return final_clip
    else:
        return silent_video_file_path

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("--npy_path", default="res_2_scott_0_1_1.npz", type=str)
    parser.add("--wav_path", default="2_scott_0_1_1.wav", type=str)
    parser.add("--save_dir", default="outputs/render", type=str)
    parser.add("--render_video_fps", default=30, type=int)
    parser.add("--render_video_width", default=1920, type=int)
    parser.add("--render_video_height", default=720, type=int)
    parser.add("--render_tmp_img_filetype", default="bmp", type=str)
    cpu_cores = os.cpu_count() if os.cpu_count() is not None else 1
    default_concurrent = max(1, cpu_cores // 2)
    parser.add("--render_concurrent_num", default=default_concurrent, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    render_one_sequence_with_face(
        args.npy_path, 
        args.save_dir,
        args.wav_path,
        "./pretrained/smplx_models/", 
        args = args,
    )
