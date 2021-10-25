import os
import cv2
from tqdm import tqdm
import numpy as np
import copy


def extract_frames_to_dataset(video_path):
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('FPS: {}'.format(fps))

    isOpened = vc.isOpened()
    frame_count = 0
    frame_list = []

    while(isOpened):
        flag, frame = vc.read()
        if flag == False:
            break

        frame_count += 1
        frame_list.append(frame)

    print('Total frame: {}'.format(frame_count))
    print('Select the first 1801 frames as dataset')
    vc.release()
    frame_list = frame_list[:1801]

    for i in range(900):
        save_image_path = '../data/video_frames_720p/' + '%04d/'%i
        if not os.path.exists(save_image_path):
            try:
                os.mkdir(save_image_path)
            except OSError:
                pass
        cv2.imwrite(save_image_path+'im1.png', frame_list[2*i])
        cv2.imwrite(save_image_path+'im2.png', frame_list[2*i+1])
        cv2.imwrite(save_image_path+'im3.png', frame_list[2*i+2])
    # with open('../data/tri_videoframes.txt', 'w') as f:
    #     for i in range(900):
    #         f.write('%04d\n'%i)


    FPS = 15
    size = (width, height)
    video = cv2.VideoWriter("../data/video/1917_720p_15fps.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), FPS, size)
    for idx, item in enumerate(frame_list):
        if idx % 2 == 0:
            video.write(item)
    video.release()

def interpolated_frames2video(file_list):
    FPS = 30
    size = (1280, 720)
    with open(file_list, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    triplet_file_list = ['../data/video_frames_720p/'+f for f in lines]

    video = cv2.VideoWriter("../data/video/ipssim-20_1917_720p_30fps.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), FPS, size)
    pbar = tqdm(total=len(triplet_file_list), desc='', ascii=True, ncols=10)
    for idx, item in enumerate(triplet_file_list):
        if idx == 0:
            im1 = cv2.resize(cv2.imread(item+'/im1.png'), size)
            im2 = cv2.resize(cv2.imread(item+'/imi_ssim_20epoch.png'), size)
            im3 = cv2.resize(cv2.imread(item+'/im3.png'), size)
            video.write(im1)
            video.write(im2)
            video.write(im3)
        else:
            im2 = cv2.resize(cv2.imread(item+'/imi_ssim_20epoch.png'), size)
            im3 = cv2.resize(cv2.imread(item+'/im3.png'), size)
            video.write(im2)
            video.write(im3)
        pbar.update(1)
    video.release()
    pbar.close()

def catimages2video(file_list):
    FPS = 30
    size = (640, 360)
    video_size = (1280, 720)
    with open(file_list, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    triplet_file_list = ['../data/video_frames_720p/'+f for f in lines]
    video = cv2.VideoWriter("../data/video/final_compare_1917_720p_30fps.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), FPS, video_size)
    pbar = tqdm(total=len(triplet_file_list), desc='', ascii=True, ncols=10)
    for idx, item in enumerate(triplet_file_list):
        if idx == 0:
            im1 = cv2.resize(cv2.imread(item+'/im1.png'), size)
            im2_1 = cv2.resize(cv2.imread(item+'/imi_weights_l162_model_epoch060.png'), size)
            im2_2 = cv2.resize(cv2.imread(item+'/imi_weights_ssim_b62_model_epoch060.png'), size)
            im2_3 = cv2.resize(cv2.imread(item+'/imi_vgg_20epoch.png'), size)
            im3 = cv2.resize(cv2.imread(item+'/im3.png'), size)

            im_cat_1 = cat_images(im1, im1, im1, im1)
            im_cat_2 = cat_images(im1, im2_1, im2_2, im2_3)
            im_cat_3 = cat_images(im3, im3, im3, im3)
            video.write(im_cat_1)
            video.write(im_cat_2)
            video.write(im_cat_3)
        else:
            im2_1 = cv2.resize(cv2.imread(item+'/im1.png'), size)
            im2_2 = cv2.resize(cv2.imread(item+'/imi_weights_l162_model_epoch060.png'), size)
            im2_3 = cv2.resize(cv2.imread(item+'/imi_weights_ssim_b62_model_epoch060.png'), size)
            im2_4 = cv2.resize(cv2.imread(item+'/imi_vgg_20epoch.png'), size)

            im3 = cv2.resize(cv2.imread(item+'/im3.png'), size)
            im_cat2 = cat_images(im2_1, im2_2, im2_3, im2_4)
            im_cat3 = cat_images(im3, im3, im3, im3)
            video.write(im_cat2)
            video.write(im_cat3)
        # im1 = cv2.resize(cv2.imread(item+'/im1.png'), size)
        # cv2.imwrite('test.png', cat_images(im1, im1, im1, im1))

        pbar.update(1)

    video.release()
    pbar.close()

def cat_images(im1, im2, im3, im4):
    '''
    im1, im2,
    im3, im4
    '''
    im1 = copy.deepcopy(im1)
    im2 = copy.deepcopy(im2)
    im3 = copy.deepcopy(im3)
    im4 = copy.deepcopy(im4)
    text = ["15 FPS (Duplicated to 30 FPS)", "30 FPS(L1)", "30 FPS(Lssim_b)", "30 FPS(Lvgg)"]
    text_size = []
    for i in range(4):
        text_size.append(cv2.getTextSize(text[i], cv2.FONT_HERSHEY_SIMPLEX, 1, 2))

    im1 = cv2.putText(im1, text[0], (im1.shape[1]//2-text_size[0][0][0]//2, im1.shape[0]//10*9-text_size[0][0][1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    im2 = cv2.putText(im2, text[1], (im2.shape[1]//2-text_size[1][0][0]//2, im2.shape[0]//10*9-text_size[1][0][1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    im3 = cv2.putText(im3, text[2], (im3.shape[1]//2-text_size[2][0][0]//2, im3.shape[0]//10*9-text_size[2][0][1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    im4 = cv2.putText(im4, text[3], (im4.shape[1]//2-text_size[3][0][0]//2, im4.shape[0]//10*9-text_size[3][0][1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    im_temp1 = np.hstack([im1, im2])
    im_temp2 = np.hstack([im3, im4])
    return np.vstack([im_temp1, im_temp2])

if __name__ == '__main__':
    # extract_frames_to_dataset('../data/video/1917_720p_30fps.avi')
    # interpolated_frames2video('../data/tri_videoframes.txt')
    catimages2video('../data/tri_videoframes.txt')
