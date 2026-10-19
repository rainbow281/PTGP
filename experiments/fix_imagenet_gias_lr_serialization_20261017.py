from pathlib import Path
root=Path('/home/yanch/GMM/experiments/imagenet_gias_stable_20261017')
matrix=root/'matrix.py'
src=matrix.read_text()
old="for k,v in c.items():args+=['--set',k+'='+json.dumps(v)]"
new="for k,v in c.items():args+=['--set',k+'='+(format(v,'.8f') if k=='gias_lr' else json.dumps(v))]"
if src.count(old)!=1:
    raise RuntimeError('Expected one argument serialization location')
matrix.write_text(src.replace(old,new))
print('GIAS LR CLI token: 0.00001000')
