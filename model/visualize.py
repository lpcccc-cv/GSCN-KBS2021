import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# def draw_features(width,height,x,savename,dpi):
#     tic=time.time()
#     fig = plt.figure(figsize=(8, 8))
#     fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
#     for i in range(width*height):
#         plt.subplot(height,width, i + 1)
#         plt.axis('off')
#         # plt.tight_layout()
#         img = x[0, i, :, :]
#         pmin = np.min(img)
#         pmax = np.max(img)
#         img = (img - pmin) / (pmax - pmin + 0.000001)  #归一化到0-1之间
#         plt.imshow(img, cmap='gray')
#         print("{}/{}".format(i,width*height))
#     fig.savefig(savename, dpi=dpi)
#     fig.clf()
#     plt.close()
#     print("time:{}".format(time.time()-tic))

# draw_features(8,8,x.cpu().numpy(),"./features/sa/f1_conv1.png",dpi=255)


def draw_features(after,dpi):
    # tic=time.time()
    # fig = plt.figure(figsize=(8, 8))
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    # for i in range(width*height):
    #     plt.subplot(height,width, i + 1)
    #     plt.axis('off')
    #     # plt.tight_layout()
    #     img = x[0, i, :, :]
    #     pmin = np.min(img)
    #     pmax = np.max(img)
    #     img = (img - pmin) / (pmax - pmin + 0.000001)  #归一化到0-1之间
    #     plt.imshow(img, cmap='gray')
    #     print("{}/{}".format(i,width*height))
    # fig.savefig(savename, dpi=dpi)
    # fig.clf()
    # plt.close()
    # print("time:{}".format(time.time()-tic))
    # print(11111111111111111)
    fig = plt.figure()
    # data1 = before[0, :, :, :].mean(0)
    # data2 = mid[0, :, :, :].mean(0)
    data3 = after[0, :, :, :].mean(0)
    # pmin = np.min(data3)
    # pmax = np.max(data3)
    # data3 = (data3 - pmin) / (pmax - pmin + 0.000001) * 255 # 归一化到0-1之间
    ax = fig.add_subplot(111)
    im = ax.imshow(data3, cmap=plt.get_cmap('hot_r'))
    plt.colorbar(im)
    # ax = fig.add_subplot(132)
    # im = ax.imshow(data2, cmap=plt.get_cmap('hot_r'), vmin=-0.2, vmax=0.5)
    # # plt.colorbar(im, shrink = 0.5)
    # ax = fig.add_subplot(133)
    # im = ax.imshow(data3, cmap=plt.get_cmap('hot_r'), vmin=-0.2, vmax=0.5)
    # plt.colorbar(im, shrink = 0.5)
    plt.savefig('/home/shiyanshi/项目代码/雷鹏程/遥感论文/遥感论文实验/src/tail2.png')
    # print('****************')
    # print(data1[0])
    # print(data2[0])
    # print(data3[0])
