# coding=utf-8
import os, sys


def scan_image_tree_get_relative_path(root_path, selected_by=None, end_withs=None, is_save_relative=True):
    ''' 索引包含指定格式文件的地址或者相对地址
    :param root_path:
    :param relative_flag:
    :param selected_by: default None
    :param end_withs: list or str,default None
    :return:
    '''

    def _scan_image_tree(dir_path, pth_list, relative_path=None, is_save_relative=True):
        if relative_path is None:
            this_folder = dir_path
        else:
            this_folder = os.path.join(dir_path, relative_path)
        for i, name in enumerate(os.listdir(this_folder)):
            if relative_path is None:
                temp = name
            else:
                temp = os.path.join(relative_path, name)

            full_path = os.path.join(this_folder, name)
            if os.path.isdir(full_path):
                _scan_image_tree(dir_path, pth_list, temp, is_save_relative=is_save_relative)
            else:
                if is_save_relative:
                    pth = temp
                else:
                    pth = full_path

                if end_withs is None:
                    if selected_by is not None or selected_by not in pth:
                        # if select_by is None or select_by in path
                        pth_list += [ pth ]
                        # if is_save_relative:
                        #     pth_list += [ pth ]
                        # else:
                        #     pth_list += [ full_path ]
                elif isinstance(end_withs, list):
                    temp = [ True for e in end_withs if pth.lower().endswith(e) ]
                    if len(temp) > 0:
                        if selected_by is None or selected_by in pth:
                            pth_list += [ pth ]
                            # if is_save_relative:
                            #     pth_list += [ pth ]
                            # else:
                            #     pth_list += [ full_path ]

                if len(pth_list) % 100 == 0:
                    sys.stdout.flush()
                    sys.stdout.write('\r #img of scan: %d' % (len(pth_list)))

    list_output = [ ]
    _scan_image_tree(root_path, list_output, is_save_relative=is_save_relative)
    sys.stdout.write('\n')
    return list_output


def scan_folder_tree_get_relative_path(root_path, selected_by=None, end_withs=None, is_save_relative=True):
    '''  索引包含指定格式文件的上级文件夹地址或者相对地址
    :param root_path:
    :param relative_flag:
    :param selected_by: default None
    :param end_withs: list or str,default None
    :return:
    '''

    def _scan_folder_tree(dir_path, pth_list, relative_path=None, is_save_relative=True):
        if relative_path is None:
            this_folder = dir_path
        else:
            this_folder = os.path.join(dir_path, relative_path)

        for i, name in enumerate(os.listdir(this_folder)):
            if relative_path is None:
                temp = name
            else:
                temp = os.path.join(relative_path, name)

            full_path = os.path.join(this_folder, name)
            if os.path.isdir(full_path):
                _scan_folder_tree(dir_path, pth_list, temp, is_save_relative=is_save_relative)
            else:
                if is_save_relative:
                    pth = temp
                else:
                    pth = full_path

                if end_withs is None:
                    # all format file
                    if selected_by is not None or selected_by not in pth:
                        # if select_by is None or select_by in path
                        pth_list += [ os.path.dirname(pth) ]
                        return
                elif isinstance(end_withs, list):
                    # if contain the endswith format files
                    temp = [ True for e in end_withs if pth.lower().endswith(e) ]
                    if len(temp) > 0:
                        # contains the file
                        if selected_by is None or selected_by in pth:
                            # if select_by is None or select_by in path
                            pth_list += [ os.path.dirname(pth) ]
                            return

                if len(pth_list) % 100 == 0:
                    sys.stdout.flush()
                    sys.stdout.write('\r #img of scan: %d' % (len(pth_list)))

    list_output = [ ]
    _scan_folder_tree(root_path, list_output, is_save_relative=is_save_relative)
    sys.stdout.write('\n')
    return list_output



def test():
    path = r'E:\datasets\110data\video'
    dat = scan_image_tree_get_relative_path(path, selected_by="F", end_withs=[ '.mp4', '.avi' ])
    print(len(dat))
    dat = scan_image_tree_get_relative_path(path, selected_by="F", end_withs=[ '.mp4', '.avi' ], is_save_relative=False)
    print(len(dat))
    dat = scan_folder_tree_get_relative_path(path, selected_by="F", end_withs=[ '.mp4', '.avi' ])
    print(len(dat))
    num = 0
    for e in dat:
        num += len(os.listdir(os.path.join(path, e)))
    print(num)
    dat = scan_image_tree_get_relative_path(path, selected_by="M", end_withs=[ '.mp4', '.avi' ])
    print(len(dat))
    dat = scan_image_tree_get_relative_path(path, selected_by="M", end_withs=[ '.mp4', '.avi' ], is_save_relative=False)
    print(len(dat))
    dat = scan_folder_tree_get_relative_path(path, selected_by="M", end_withs=[ '.mp4', '.avi' ])
    print(len(dat))
    dat = scan_folder_tree_get_relative_path(path, selected_by="M", end_withs=[ '.mp4', '.avi' ],
                                             is_save_relative=False)
    print(len(dat))
    num = 0
    for e in dat:
        num += len(os.listdir(os.path.join(path, e)))
    print(num)
    pass

def static_146():
    path = r'E:\datasets\110data\video'
    # path = r'E:\datasets\110data\video\12'
    # path = r'E:\datasets\110data\video\1'
    # path = r'E:\datasets\110data\video\2'
    # path = r'E:\datasets\110data\1'

    Female = scan_folder_tree_get_relative_path(path, selected_by="-F-",
                                                end_withs=[ '.mp4', '.avi', '.wmv', '.3gp', '.mov', '.m4v', '.flv',
                                                            '.mkv', '.vob' ],
                                                is_save_relative=False)
    print("Female num:{}".format(len(Female)))
    Male = scan_folder_tree_get_relative_path(path, selected_by="-M-",
                                              end_withs=[ '.mp4', '.avi', '.wmv', '.3gp', '.mov', '.m4v', '.flv',
                                                          '.mkv', '.vob' ],
                                              is_save_relative=False)
    print("Male num:{}".format(len(Male)))
    import numpy as np

    print(len(Female) + len(Male))

    path = r'E:\datasets\110data\video\2\2'
    count = 0
    for e in os.listdir(path):
        if len(os.listdir(os.path.join(path, e))) == 0:
            continue
        count += 1

    print(count)



    data = Female + Male
    count_dat = np.zeros(100 // 5)
    count_dat_video = np.zeros(100 // 5)
    count_data_folder = [ [ ] for i in range(len(count_dat)) ]
    for e in data:
        folder_basename = os.path.basename(e)
        age = int(folder_basename.split('-')[ -1 ])
        index = int((age - 1) / 5)
        count_dat[ index ] += 1
        count_data_folder[ index ].append(e)
        video_num = len(os.listdir(e))
        count_dat_video[ index ] += video_num

    print(count_dat)
    print(count_dat_video)

    print(sum(count_dat))
    print(sum(count_dat_video))

    for i in range(len(count_dat)):
        if count_dat[ i ] > 0:
            print(count_data_folder[ i ])
            np.savetxt("{}.txt".format(5 * i + 1), np.array(count_data_folder[i],dtype=str),fmt='%s')
####### test
if __name__ == '__main__':
    pass