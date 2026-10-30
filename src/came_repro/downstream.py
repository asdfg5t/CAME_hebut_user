"""Frozen node probes, KG relation classification, and few-shot evaluation."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .author_loader import ROOT, fingerprint
from .prepared_data import load_prepared, prepared_directory
from .stream_inference import extract_full_embeddings
from .train import make_model
from .experiment import Experiment, seed_everything, write_json, atomic_checkpoint, rng_state
from .task_metrics import support_query_episode, prototype_predictions
from .pause_control import TrainingPaused, pause_requested


class ProbeInput:
    def __init__(self, embeddings, labels, indices=None, pairs=None):
        self.embeddings = embeddings
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64) if indices is not None else None
        self.pairs = np.asarray(pairs, dtype=np.int64) if pairs is not None else None
        self.width = embeddings.shape[1]*(2 if pairs is not None else 1)

    def __len__(self):
        return len(self.labels)

    def batch(self, index):
        if self.pairs is not None:
            selected = self.pairs[index]
            features = np.concatenate([np.asarray(self.embeddings[selected[:, 0]]), np.asarray(self.embeddings[selected[:, 1]])], axis=1)
        else:
            features = np.asarray(self.embeddings[self.indices[index]])
        return torch.from_numpy(np.array(features, dtype=np.float32, copy=True)).cuda(), torch.from_numpy(self.labels[index].copy()).long().cuda()


@torch.no_grad()
def score_probe(classifier, data, batch_size=4096, save_predictions=False):
    classifier.eval()
    correct, loss_sum, predictions = 0, 0., []
    for start in range(0, len(data), batch_size):
        features, labels = data.batch(np.arange(start, min(start+batch_size, len(data))))
        logits = classifier(features)
        loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
        pred = logits.argmax(1)
        correct += int((pred == labels).sum())
        if save_predictions:
            predictions.append(pred.cpu().numpy())
    return correct/len(data), loss_sum/len(data), np.concatenate(predictions) if save_predictions else None


def train_linear_probe(train, validation, test, classes, epochs, patience, seed, experiment, lr=.05, batch_size=0, selection_criterion='accuracy', resume=None, evaluate_test=True):
    seed_everything(seed)
    classifier = torch.nn.Linear(train.width, classes).cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    random = np.random.default_rng(seed)
    best, best_epoch, stale, rows = -1., 0, 0, []
    best_state = None
    best_score = float('-inf')
    selected_loss = None
    first_epoch=1
    if resume:
        from .head_resume import restore_head
        previous=restore_head(resume,'came_linear_probe_v1',experiment,classifier,optimizer,random)
        first_epoch=previous['next_epoch']
        best_state,best_epoch,best_score,stale=previous['best_state'],previous['selected_epoch'],previous['best_score'],previous['stale']
        best,selected_loss=previous['selection_accuracy'],previous['selection_loss']
        rows=list(previous['history_rows'])
    full_batch = batch_size == 0
    if full_batch:
        required = len(train)*train.width*4 + len(train)*classes*12
        if required + 512*1024**2 > torch.cuda.mem_get_info()[0]:
            raise RuntimeError('Full-batch linear probe exceeds GPU budget; select and record a mini-batch adaptation explicitly.')
        cached_features, cached_labels = train.batch(np.arange(len(train)))
    epoch = first_epoch - 1
    for epoch in range(first_epoch, first_epoch if stale >= patience else epochs+1):
        classifier.train()
        total = 0.
        order = np.arange(len(train)) if full_batch else random.permutation(len(train))
        effective_batch = len(train) if full_batch else batch_size
        for start in range(0, len(train), effective_batch):
            features, labels = (cached_features, cached_labels) if full_batch else train.batch(order[start:start+effective_batch])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(features), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite downstream loss')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(labels)
        accuracy, val_loss, _ = score_probe(classifier, validation)
        row = {'step': epoch, 'train_loss': total/len(train), 'val_loss': val_loss, 'val_accuracy': accuracy, 'lr': lr}
        experiment.metric(row)
        rows.append(row)
        score = accuracy if selection_criterion == 'accuracy' else -val_loss
        if score > best_score:
            best_score, selected_loss = score, val_loss
            best, best_epoch, stale = accuracy, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
        else:
            stale += 1
        if epoch % 25 == 0 or epoch == 1:
            experiment.logger.info('probe epoch=%d train=%.6f selection_accuracy=%.6f selected_epoch=%d', epoch, total/len(train), accuracy, best_epoch)
        if epoch == 1 or epoch % 25 == 0 or stale >= patience or epoch == epochs or pause_requested():
            atomic_checkpoint(experiment.checkpoint_directory/'probe_last.pt',
                {'schema':'came_linear_probe_v1','classifier':classifier.state_dict(),'optimizer':optimizer.state_dict(),
                 'rng':rng_state(),'sampling_generator':random.bit_generator.state,'completed_epochs':epoch,
                 'next_epoch':epoch+1,'cursor_policy':'Epoch boundary; all train rows consumed',
                 'best_state':best_state,'selected_epoch':best_epoch,'best_score':best_score,'stale':stale,
                 'selection_accuracy':best,'selection_loss':selected_loss,'history_rows':rows,
                 'config':experiment.config,'data_audit':experiment.data_audit})
        if pause_requested():
            raise TrainingPaused('Linear probe saved at epoch boundary; epoch=' + str(epoch))
        if stale >= patience:
            break
    classifier.load_state_dict(best_state)
    test_accuracy, test_loss = None, None
    if evaluate_test:
        test_accuracy, test_loss, predictions = score_probe(classifier, test, save_predictions=True)
    atomic_checkpoint(experiment.checkpoint_directory / 'probe_best.pt', {'classifier': best_state, 'selected_epoch': best_epoch, 'selection_accuracy': best, 'selection_loss': selected_loss, 'selection_criterion': selection_criterion})
    if evaluate_test:
        np.savez_compressed(experiment.directory / 'test_predictions.npz', prediction=predictions, labels=test.labels,
                            node_ids=test.indices if test.indices is not None else np.empty(0,dtype=np.int64),
                            pairs=test.pairs if test.pairs is not None else np.empty((0,2),dtype=np.int64))
    else:
        selection_accuracy, selection_loss, predictions = score_probe(classifier, validation, save_predictions=True)
        if abs(selection_accuracy-best) > 1e-7:
            raise ValueError('Saved selection predictions disagree with the selected validation accuracy')
        np.savez_compressed(experiment.directory / 'selection_predictions.npz', prediction=predictions,
                            labels=validation.labels, node_ids=validation.indices)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    run_config=json.loads((experiment.directory/'config.json').read_text())
    evaluation=run_config.get('evaluation',{})
    selection_split=evaluation.get('selection_split','official_validation')
    for key in ('train_loss','val_loss'):
        label='train_loss' if key=='train_loss' else 'stopping_loss' if selection_split=='official_stopping_mask' else 'validation_loss'
        axes[0].plot([row['step'] for row in rows],[row[key] for row in rows],label=label,marker='o' if len(rows)==1 else None)
    axes[0].set(xlabel='Probe epoch',ylabel='Cross entropy (natural log)',title='Frozen encoder linear probe')
    axes[0].legend()
    axes[1].plot([row['step'] for row in rows],[row['val_accuracy'] for row in rows],marker='o' if len(rows)==1 else None)
    axes[1].axvline(best_epoch,color='black',linestyle='--')
    axes[1].set(xlabel='Probe epoch',ylabel='Selection-set accuracy (fraction)',title='Model selection; test held out')
    test_policy = 'test once after selection' if evaluate_test else 'validation-only diagnostic; test not evaluated'
    fig.text(.02,.01,f'Source: {evaluation.get("dataset","official dataset")}, split={evaluation.get("split_id",0)}, seed={seed}; selection={selection_split}; {test_policy}.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(experiment.directory / 'figures/probe_curves.png',dpi=180)
    plt.close(fig)
    return {'seed':seed,'selected_epoch':best_epoch,'epochs_executed':len(rows),'validation_accuracy':best,
            'test_accuracy':test_accuracy,'test_loss':test_loss,'test_evaluations':int(evaluate_test),
            'selection_criterion':selection_criterion,'selection_loss':selected_loss,
            'optimizer':'Adam','lr':lr,'training_batch_size':batch_size,'max_epochs':epochs,'patience':patience,
            'assumptions':'Optimizer and early-stopping patience reconstructed; batch_size=0 means full-batch. Nonzero mini-batches are a declared adaptation.'}


def node_inputs(data, embeddings, split_id):
    if data.audit['dataset'] == 'wiki-cs':
        train = np.flatnonzero(data.splits['train_masks'][split_id])
        if not 0 <= split_id < len(data.splits['train_masks']):
            raise ValueError('Wiki-CS split_id must select one of the 20 official splits')
        validation = np.flatnonzero(data.splits['stopping_masks'][split_id])
        test = np.flatnonzero(data.splits['test_mask'])
    else:
        train, validation, test = (np.asarray(data.splits[key]) for key in ('train','val','test'))
    sets = [set(value.tolist()) for value in (train,validation,test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Official labeled train/validation/test node sets overlap')
    result=[]
    for index in (train,validation,test):
        index=index[np.asarray(data.labels[index])>=0]
        result.append(ProbeInput(embeddings,np.asarray(data.labels[index]),indices=index))
    return result


def kg_inputs(dataset, embeddings):
    directory=prepared_directory(dataset)
    result=[]
    for split in ('train','valid','test'):
        triples=np.load(directory/(split+'_triples.npy'),mmap_mode='r')
        result.append(ProbeInput(embeddings,triples[:,1],pairs=triples[:,[0,2]]))
    classes=json.loads((directory/'manifest.json').read_text())['relations']
    return result,classes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--resume-head',type=Path)
    parser.add_argument('--representation',choices=['came','clip_text','clip_image'],default='came')
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--task',choices=['node','relation','few_shot'],required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--split-id',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=2000)
    parser.add_argument('--patience',type=int,default=100)
    parser.add_argument('--inference-batch-size',type=int,default=64)
    parser.add_argument('--probe-batch-size',type=int,default=0)
    args=parser.parse_args()
    if args.resume_head and args.task=='few_shot':
        raise ValueError('Few-shot evaluation has no trainable head to resume')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    seed_everything(args.seed)
    storage=json.loads((ROOT/'configs/storage.json').read_text())
    if args.representation=='came':
        if args.checkpoint is None:
            raise ValueError('CAME evaluation requires a joint checkpoint')
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        config=saved['config']
        if config['stage']!='paper_reconstruction_with_declared_assumptions':
            raise ValueError('This task entry expects a joint-paper reconstruction checkpoint')
        expected_corpus=['ogbn-products','books-nc','cloth-copurchase'] if args.task=='few_shot' else ['ogbn-products','books-lp','cloth-copurchase']
        if config['datasets']!=expected_corpus:
            raise ValueError('Checkpoint corpus does not match the requested paper task')
        complete=saved.get('completed_epochs',saved.get('schedule',{}).get('epoch'))==8
    else:
        config={'stage':'frozen_clip_baseline','storage':storage,'deviations':['Frozen official CLIP features; optimizer and patience are disclosed reconstruction choices.']}
        saved=None
        complete=None
    data=load_prepared(args.dataset,require_verified=True,build_graph=args.representation=='came')
    if args.representation=='clip_image' and data.features.shape[1]==512:
        raise ValueError('Paper reports no CLIP-image baseline for text-only graphs')
    run_config={**config,'name':args.representation+'_'+args.task+'_'+args.dataset.replace('-','_').lower(), 'seed':args.seed,
                'evaluation':vars(args)|{'checkpoint':str(args.checkpoint) if args.checkpoint else None,'complete_pretraining':complete}}
    run_config['evaluation']={key:str(value) if isinstance(value,Path) else value for key,value in run_config['evaluation'].items()}
    run_config['evaluation']['encoder_sha256']=fingerprint(args.checkpoint) if args.checkpoint else None
    wiki_node=args.dataset=='wiki-cs' and args.task=='node'
    run_config['evaluation']['selection_split']='official_stopping_mask' if wiki_node else 'official_validation'
    run_config['evaluation']['selection_criterion']='loss' if wiki_node else 'accuracy'
    experiment=Experiment(run_config,data.audit,args.resume_head or args.checkpoint)
    try:
        if args.representation=='came':
            model=make_model(config)
            model.load_state_dict(saved['model'])
            del saved
            checkpoint_hash=fingerprint(args.checkpoint)
            cache=Path(storage['embedding_cache_root'])/checkpoint_hash[:20]/args.dataset
            embeddings=extract_full_embeddings(model,data,cache,checkpoint_hash,args.inference_batch_size,experiment.logger)
            del model
            torch.cuda.empty_cache()
        else:
            if isinstance(data.features,torch.Tensor):
                embeddings=data.features[:,512:].numpy() if args.representation=='clip_image' else data.features[:,:512].numpy()
            else:
                embeddings=data.features
            cache=None
        if args.task=='few_shot':
            candidates=np.unique(np.concatenate([np.asarray(value).flatten() for key,value in data.splits.items() if key in ('train','val','test')])) if args.dataset!='wiki-cs' else np.arange(data.audit['nodes'])
            records=[]
            for shot in (1,3,5):
                chosen,support,query=support_query_episode(data.labels,candidates,5,shot,args.seed)
                prediction=prototype_predictions(embeddings,chosen,support,query)
                accuracy=float(np.mean(prediction==np.asarray(data.labels[query])))
                np.savez_compressed(experiment.directory/f'episode_{shot}shot.npz',classes=chosen,support_ids=support,query_ids=query,prediction=prediction)
                records.append({'shot':shot,'way':5,'seed':args.seed,'query_count':len(query),'accuracy':accuracy})
                experiment.metric({'step':shot,'way':5,'query_count':len(query),'accuracy':accuracy})
                experiment.logger.info('Few-shot way=5 shot=%d queries=%d accuracy=%.6f',shot,len(query),accuracy)
            fig, axis = plt.subplots(figsize=(6,4))
            axis.plot([row['shot'] for row in records],[row['accuracy'] for row in records],marker='o')
            axis.set(xlabel='Support nodes per class',ylabel='Query accuracy (fraction)',xticks=[1,3,5],ylim=(0,1),title='5-way frozen-representation evaluation')
            fig.text(.02,.01,f'Source: {args.dataset}; seed={args.seed}; cosine prototypes; disjoint support/query.',fontsize=8)
            fig.tight_layout(rect=(0,.08,1,1))
            fig.savefig(experiment.directory/'figures/few_shot_accuracy.png',dpi=180)
            plt.close(fig)
            summary={'task':'few_shot','episodes':records,'matching':'cosine prototype; reconstructed distance choice',
                     'query_policy':'All remaining officially labeled nodes of the sampled classes; support/query disjoint'}
        else:
            if args.task=='relation':
                inputs,classes=kg_inputs(args.dataset,embeddings)
            else:
                inputs=node_inputs(data,embeddings,args.split_id)
                classes=int(np.max(data.labels))+1
            summary=train_linear_probe(*inputs,classes,args.epochs,args.patience,args.seed,experiment,batch_size=args.probe_batch_size,selection_criterion='loss' if wiki_node else 'accuracy',resume=args.resume_head)
            if wiki_node:
                summary['stopping_accuracy']=summary.pop('validation_accuracy')
                selected=torch.load(experiment.checkpoint_directory/'probe_best.pt',map_location='cpu',weights_only=False)
                classifier=torch.nn.Linear(inputs[0].width,classes).cuda()
                classifier.load_state_dict(selected['classifier'])
                valid_ids=np.flatnonzero(data.splits['val_masks'][args.split_id])
                valid=ProbeInput(embeddings,data.labels[valid_ids],indices=valid_ids)
                summary['validation_accuracy'],summary['validation_loss'],_=score_probe(classifier,valid)
                summary['selection_split']='official_stopping_mask'
            summary['task']=args.task
        summary.update(dataset=args.dataset,representation=args.representation,encoder_checkpoint=str(args.checkpoint) if args.checkpoint else None,encoder_frozen=True,
                       pretraining_complete=run_config['evaluation']['complete_pretraining'],paper_reproduction_complete=False,
                       feature_cache=str(cache) if cache else 'Direct view of existing verified CLIP features; no duplicate cache',deviations=config['deviations'])
        write_json(experiment.directory/'summary.json',summary)
        experiment.status('completed_downstream_task')
        print(f'RUN_DIR={experiment.directory}',flush=True)
    except BaseException as error:
        if isinstance(error, TrainingPaused):
            experiment.logger.info('Safe pause: %s', error)
        else:
            experiment.logger.exception('Downstream task failed')
        experiment.status('paused' if isinstance(error, TrainingPaused) else 'failed',error=str(error))
        raise
    finally:
        experiment.close()


if __name__=='__main__':
    main()
