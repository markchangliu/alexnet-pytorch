import multiprocessing as mp
import os
import shutil
from typing import Union, Dict, Any, List, Callable, Tuple

import cv2
import numpy as np


def load_tiny_imagenet_cats(
    data_root: Union[str, os.PathLike]
) -> Dict[str, int]:
    cat_name_id_dict = {}

    dirnames = os.listdir(data_root)
    dirnames = [d for d in dirnames if os.path.isdir(os.path.join(data_root, d))]
    dirnames.sort()
    cat_name_id_dict = {d: i for i, d in enumerate(dirnames)}

    return cat_name_id_dict

def mp_func(
    dispatch_func: Callable[[mp.Queue], None],
    worker_func: Callable[[mp.Queue, mp.Queue], None],
    agg_func: Callable[[mp.Queue], Any],
    num_workers: int,
) -> Any:
    """
    Args
    - `dispatch_func`: `Callable[[mp.Queue], None]`
        - `args`: 
            - `in_queue`, `mp.Queue`
        - PUT "STOP signal to `in_queue`
    - `worker_func`: `Callable[[mp.Queue, mp.Queue], None]`
        - `args`: 
            - `in_queue`, `mp.Queue`
            - `out_queue`, `mp.Queue`
        - GET `STOP` from `in_queue`
        - PUT `"ERROR: ..."` to `out_queue`
        - PUT `"DONE"` to `out_queue`
    - `agg_func`: `Callable[[mp.Queue], Any]`
        - `args`: 
            - `out_queue`, `mp.Queue`
            - `num_workers`: `int`
        - GET `ERROR` to `out_queue`
        - GET `DONE` to `out_queue`
    """
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    procs = []

    for i in range(num_workers):
        proc = mp.Process(
            target = worker_func, 
            args = (in_queue, out_queue)
        )
        proc.start()
        procs.append(proc)
    
    dispatch_func(in_queue)

    for i in range(num_workers):
        in_queue.put("STOP")

    agg_res = agg_func(out_queue, num_workers)

    for proc in procs:
        proc.terminate()
        proc.join()

    in_queue.close()
    out_queue.close()

    return agg_res

def get_channel_mean_std(
    data_root: Union[str, os.PathLike],
    num_workers: int
) -> Tuple[np.ndarray, np.ndarray]:
    def worker_func(
        in_queue: mp.Queue,
        out_queue: mp.Queue
    ) -> None:
        for img_p in iter(in_queue.get, "STOP"):
            try:
                img = cv2.imread(img_p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                channel_mean = np.mean(img, axis = (0, 1))
                channel_std = np.std(img, axis = (0, 1))
                out_queue.put((channel_mean, channel_std))
            except Exception as e:
                out_queue.put(f"ERROR: {e}")
        
        out_queue.put("DONE")
    
    def dispatch_func(
        in_queue: mp.Queue
    ) -> None:
        for root, dirnames, filenames in os.walk(data_root):
            for filename in filenames:
                if not filename.endswith(".JPEG"):
                    continue

                img_p = os.path.join(root, filename)
                in_queue.put(img_p)
    
    def agg_func(
        out_queue: mp.Queue,
        num_workers: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_stop_workers = 0
        channel_mean = np.zeros(3)
        channel_std = np.zeros(3)
        k = 1

        while num_stop_workers < num_workers:
            res = out_queue.get()

            if isinstance(res, str) and res.startswith("ERROR"):
                print(res)
                continue
            elif isinstance(res, str) and res.startswith("DONE"):
                num_stop_workers += 1
                continue
            
            channel_mean_k, channel_std_k = res
            channel_mean = channel_mean + (channel_mean_k - channel_mean) / k
            channel_std = channel_std + (channel_std_k - channel_std) / k
            k += 1

            if k % 10000 == 0:
                print(f"complete {k} images")

        return channel_mean, channel_std
    
    channel_mean, channel_std = mp_func(
        dispatch_func, worker_func, agg_func, num_workers
    )

    print(f"channel_mean: {channel_mean}")
    print(f"channel_std: {channel_std}")

    return channel_mean, channel_std

def get_eigenval_eigenvec(
    data_root: Union[str, os.PathLike],
    channel_mean: np.ndarray,
    num_workers: int
) -> Tuple[np.ndarray, np.ndarray]:
    def worker_func(
        in_queue: mp.Queue,
        out_queue: mp.Queue
    ) -> None:
        for img_p in iter(in_queue.get, "STOP"):
            try: 
                img = cv2.imread(img_p)
                img_h, img_w = img.shape[:2]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                img_centered = img - channel_mean.reshape(1, 1, 3)
                img_centered_T = img_centered.reshape(-1, 3)
                img_centered = np.transpose(img_centered_T, (1, 0))
                sigma = np.matmul(img_centered, img_centered_T) / (img_h * img_w - 1)
                out_queue.put(sigma)
            except Exception as e:
                out_queue.put(f"ERROR: {e}")
        
        out_queue.put("DONE")
    
    def dispatch_func(
        in_queue: mp.Queue
    ) -> None:
        for root, dirnames, filenames in os.walk(data_root):
            for filename in filenames:
                if not filename.endswith(".JPEG"):
                    continue

                img_p = os.path.join(root, filename)
                in_queue.put(img_p)
    
    def agg_func(
        out_queue: mp.Queue,
        num_workers: int
    ) -> np.ndarray:
        num_stop_workers = 0
        k = 1
        sigma = np.zeros((3, 3))

        while num_stop_workers < num_workers:
            res = out_queue.get()

            if isinstance(res, str) and res.startswith("ERROR:"):
                print(res)
                continue
            elif isinstance(res, str) and res.startswith("DONE"):
                num_stop_workers += 1
                continue

            sigma_k: np.ndarray = res
            sigma = sigma + (sigma_k - sigma) / k
            k += 1

            if k % 10000 == 0:
                print(f"complete {k} images")

        return sigma
    
    sigma = mp_func(
        dispatch_func, worker_func, agg_func, num_workers
    )

    eigenval, eigenvec = np.linalg.eig(sigma)

    print("sigma:", sigma)
    print("eigenval:", eigenval)
    print("eigenvec", eigenvec)

    return eigenval, eigenvec


def make_val_img_folder(
    val_img_dir: Union[str, os.PathLike],
    val_ann_p: Union[str, os.PathLike],
    export_dir: Union[str, os.PathLike]
) -> None:
    filenames = []
    cat_names = []
    bboxes = []

    with open(val_ann_p, "r") as f:
        for line in f:
            line = line.strip().split()
            filenames.append(line[0])
            cat_names.append(line[1])
            bbox = [int(i) for i in line[2:]]
            bboxes.append(bbox)
    
    for filename, cat_name, bbox in zip(filenames, cat_names, bboxes):
        dst_dir = os.path.join(export_dir, cat_name)
        os.makedirs(dst_dir, exist_ok = True)

        cat_box_ann_p = os.path.join(dst_dir, f"{cat_name}_boxes.txt")
        with open(cat_box_ann_p, "a") as f:
            f.write(f"{filename}\t{bbox[0]}\t{bbox[1]}\t{bbox[2]}\t{bbox[3]}\n")

        src_img_p = os.path.join(val_img_dir, filename)
        dst_img_dir = os.path.join(dst_dir, "images")
        dst_img_p = os.path.join(dst_img_dir, filename)

        os.makedirs(dst_img_dir, exist_ok = True)
        shutil.copy(src_img_p, dst_img_p)
    