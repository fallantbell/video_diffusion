import cv2
import torch

def video_tensor_to_mp4(tensor, path, step, fps=30):

    video_path = f"{path}_{step}.mp4"
    begin_path = f"{path}_{step}_begin.png"
    end_path = f"{path}_{step}_end.png"

    tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)

    height, width = tensor.shape[-2:]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    idx = 0

    for frame_tensor in tensor.unbind(dim=1):

        frame_cv = frame_tensor.cpu().numpy()
        frame_cv = cv2.cvtColor(frame_cv.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

        if idx == 0:
            cv2.imwrite(begin_path,frame_cv)
            idx+=1

        video_writer.write(frame_cv)

    cv2.imwrite(end_path,frame_cv)

    video_writer.release()
