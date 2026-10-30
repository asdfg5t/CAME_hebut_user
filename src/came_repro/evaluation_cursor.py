"""Durable deterministic evaluation outputs; restart never rescores committed rows."""
import hashlib
from pathlib import Path
import numpy as np
from .atomic_io import atomic_write_json,read_json_file


class EvaluationCursor:
    def __init__(self,checkpoint,identity,length,dtype,resume=None):
        if type(length) is not int or length<1:raise ValueError('Positive official evaluation length required')
        self.checkpoint=Path(checkpoint);self.identity=dict(identity,total=length,dtype=np.dtype(dtype).str)
        self.digest=hashlib.sha256()
        if resume:
            state=read_json_file(resume)
            if state['identity']!=self.identity:raise ValueError('Evaluation head/data/protocol identity changed')
            self.values_path=Path(state['values_path']);self.cursor=state['cursor']
            if type(self.cursor) is not int or not 0<=self.cursor<=length:raise ValueError('Evaluation cursor out of range')
        else:
            self.values_path=self.checkpoint.parent/'test_values.npy';self.cursor=0
            if self.values_path.exists():raise ValueError('Refuse to overwrite existing evaluation values')
            array=np.lib.format.open_memmap(self.values_path,mode='w+',dtype=dtype,shape=(length,));array.flush();array._mmap.close()
        array=np.load(self.values_path,mmap_mode='r')
        if array.shape!=(length,) or array.dtype!=np.dtype(dtype):raise ValueError('Evaluation storage shape/type changed')
        self.offset=array.offset;array._mmap.close();self.itemsize=np.dtype(dtype).itemsize;self.dtype=np.dtype(dtype);self.length=length
        with self.values_path.open('rb') as file:
            file.seek(self.offset);remaining=self.cursor*self.itemsize
            while remaining:
                value=file.read(min(8*2**20,remaining))
                if not value:raise ValueError('Truncated evaluation prefix')
                self.digest.update(value);remaining-=len(value)
        if resume and self.digest.hexdigest()!=state['prefix_sha256']:raise ValueError('Evaluation committed prefix changed')
        self.persist()

    def persist(self):
        atomic_write_json(self.checkpoint,dict(identity=self.identity,values_path=str(self.values_path),cursor=self.cursor,
            prefix_sha256=self.digest.hexdigest(),deterministic_eval_no_sampling_or_training_RNG=True))

    def commit(self,start,values):
        import os
        values=np.asarray(values,dtype=self.dtype)
        if start!=self.cursor or values.ndim!=1 or self.cursor+len(values)>self.length:raise ValueError('Noncontiguous evaluation cursor')
        if not np.isfinite(values).all():raise ValueError('Nonfinite evaluation output')
        blob=values.tobytes()
        with self.values_path.open('r+b') as file:
            file.seek(self.offset+self.cursor*self.itemsize)
            if file.write(blob)!=len(blob):raise OSError('Incomplete evaluation row write')
            file.flush();os.fsync(file.fileno())
        self.digest.update(blob);self.cursor+=len(values);self.persist()

    def complete_values(self):
        if self.cursor!=self.length:raise ValueError('Complete official evaluation required')
        return np.load(self.values_path,mmap_mode='r')
