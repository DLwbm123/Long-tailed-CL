"""Small new-objective reference check, no Torch, patient data or optimization."""
import json,math,time
from pathlib import Path
from reference import objective_and_grad
from geometry import matching_cells


def main():
    first = [[[.3*math.sin(1+i*3+k*2+j) for j in range(3)] for k in range(2)] for i in range(5)]
    second = [[[.4*math.cos(2+i*2+k+j*3) for j in range(3)] for k in range(2)] for i in range(5)]
    loss, grad = objective_and_grad(first, second)
    largest = 0.; calls = 1
    for view, values in enumerate((first, second)):
        for i in range(5):
            for k in range(2):
                for j in range(3):
                    old = values[i][k][j];eps=1e-6
                    values[i][k][j]=old+eps;plus=objective_and_grad(first, second)[0]
                    values[i][k][j]=old-eps;minus=objective_and_grad(first, second)[0]
                    values[i][k][j]=old;calls+=2
                    largest=max(largest,abs((plus-minus)/(2*eps)-grad[view][i][k][j]))
    minimum=196
    shifts=[(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)]
    for a in shifts:
        for b in shifts:
            x,y=matching_cells(a,b);minimum=min(minimum,len(x))
            assert len(x)==len(set(x))==len(y)==len(set(y))
            for u,v in zip(x,y):
                assert (u//14-a[0],u%14-a[1])==(v//14-b[0],v%14-b[1])
                assert 0<=u//14-a[0]<14 and 0<=u%14-a[1]<14
    record=dict(state='PASS_REFERENCE_ONLY' if largest<2e-6 else 'FAIL',checked_epoch=time.time(),
                objective=loss,max_derivative_error=largest,objective_calls=calls,objective_rows=calls*10,
                geometry_pairs=81,min_matched_cells=minimum,Torch='NOT_RUN',NIH_images=0,GPU_calls=0,optimizer_updates=0)
    (Path(__file__).parent/'local_runtime/REFERENCE_CHECK.private.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record));assert largest<2e-6 and minimum==144


if __name__=='__main__':main()
