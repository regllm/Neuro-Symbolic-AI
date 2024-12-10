from typing import Optional, Iterable, Callable

import ctypes
import multiprocessing
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from tqdm import tqdm

def multiprocess_ehs_calc(
    source_iter: Iterable[any],
    tasker: Callable[[Iterable[any], int, int], None],
    result_size: int = None,
    result_width: int = 3,
):
    result_size = result_size or len(source_iter)
    result_bytes = result_size * result_width * 8
    result_sm = SharedMemory(
        name="result_sm", create=True, size=result_bytes
    )
    result = np.ndarray(
        (result_size, result_width), dtype=np.double, buffer=result_sm.buf
    )

    def process_all(batch, cursor):
        sm = SharedMemory("result_sm")
        result = np.ndarray(
            (result_size, result_width), dtype=np.double, buffer=sm.buf
        )
        tasker(batch, cursor, result)
        sm.close()
    
    worker_count = multiprocessing.cpu_count()
    batch_size = min(10_000, result_size // worker_count)
    cursor = 0
    max_batch_seconds = None
    batch_failed = False
    total = result_size
    if total is None:
        try:
            total = len(source_iter)
        except:
            pass

    with tqdm(total=total, ascii=" >=") as pbar:
        while True:
            task_done = False
            
            if batch_failed:
                batches = cached_batches
            else:
                batches = []
                for _ in range(worker_count):
                    batch = []
                    for _ in range(batch_size):
                        try:
                            batch.append(next(source_iter))
                        except StopIteration:
                            task_done = True
                            break
                    if len(batch) > 0:
                        batches.append(batch)
            
            cached_batches = batches
            batch_failed = False
            
            total_batch_size = 0
            processes = []

            start = time.time()
            for batch in batches:
                process = multiprocessing.Process(
                    target=process_all, args=(batch, cursor)
                )
                process.start()
                processes.append(process)
                total_batch_size += len(batch)
                cursor += len(batch)
        
            for process in processes:
                if max_batch_seconds is None:
                    process.join()
                    continue
                process.join(timeout=max_batch_seconds)

                if process.is_alive():
                    # Failed to handle this batch.
                    batch_failed = True
                    cursor -= total_batch_size
                    break

            if batch_failed:
                continue

            end = time.time()
            if max_batch_seconds is None:
                duration = end - start
                max_batch_seconds = int(duration * 3)
                
            pbar.update(total_batch_size)
            
            if task_done:
                break
    
    return result, result_sm


def batch_process(
    source_iter: Iterable[any],
    batch_process: Callable[[Iterable[any], int, multiprocessing.Array], None],
    result_type: ctypes._SimpleCData = ctypes.c_float,
    result_size: int = None,
    worker_count: Optional[int] = None,
    max_batch_size: Optional[int] = None,
):
    result_array = multiprocessing.Array(
        result_type, result_size
    )

    worker_count = worker_count or multiprocessing.cpu_count()
    batch_size = min(max_batch_size or 10_000, result_size // worker_count)
    cursor = 0
    max_batch_seconds = None
    batch_failed = False
    total = result_size
    if total is None:
        try:
            total = len(source_iter)
        except:
            pass

    with tqdm(total=total, ascii=" >=") as pbar:
        while True:
            task_done = False
            
            if batch_failed:
                batches = cached_batches
            else:
                batches = []
                for _ in range(worker_count):
                    batch = []
                    for _ in range(batch_size):
                        try:
                            batch.append(next(source_iter))
                        except StopIteration:
                            task_done = True
                            break
                    if len(batch) > 0:
                        batches.append(batch)
            
            cached_batches = batches
            batch_failed = False
            
            total_batch_size = 0
            processes = []

            start = time.time()
            for batch in batches:
                process = multiprocessing.Process(
                    target=batch_process, args=(batch, cursor, result_array)
                )
                process.start()
                processes.append(process)
                total_batch_size += len(batch)
                cursor += len(batch)
        
            for process in processes:
                if max_batch_seconds is None:
                    process.join()
                    continue
                process.join(timeout=max_batch_seconds)

                if process.is_alive():
                    # Failed to handle this batch.
                    batch_failed = True
                    cursor -= total_batch_size
                    break

            if batch_failed:
                continue

            end = time.time()
            if max_batch_seconds is None:
                duration = end - start
                max_batch_seconds = int(duration * 3)
                
            pbar.update(total_batch_size)
            
            if task_done:
                break
    
    return result_array
