"""Small trust-boundary check for the official RFMiD CSV parser."""
import csv
import io
import json
import time
from prepare_rfmid_ml import DISEASES, read_labels, benchmark_labels


def check():
    begun=time.monotonic()
    columns=['ID','Disease_Risk',*DISEASES]
    def table(rows,header=columns):
        out=io.StringIO();writer=csv.writer(out);writer.writerow(header);writer.writerows(rows)
        return out.getvalue()
    good=['1','0',*(['0']*45)]
    assert read_labels(table([good]),1)[0]['ID']=='1'
    bad_cases=[(table([['../1',*good[1:]]]),1),(table([['01',*good[1:]]]),1),
               (table([['1','?',*good[2:]]]),1),(table([['1','2',*good[2:]]]),1),
               (table([good,good]),2),(table([good],columns[::-1]),1),
               (table([good]),2)]
    for text,expected in bad_cases:
        try:read_labels(text,expected)
        except (AssertionError,KeyError,TypeError,ValueError):pass
        else:raise AssertionError('Invalid labels accepted')
    row=read_labels(table([good]),1)[0]
    assert benchmark_labels(row)==[1,*([0]*28)]
    row.update(Disease_Risk='1',DR='1',MCA='1',CL='1')
    mapped=benchmark_labels(row)
    assert mapped[0]==0 and mapped[1]==1 and mapped[-1]==1 and sum(mapped)==2
    row['DR']='0'
    assert sum(benchmark_labels(row))==1 and benchmark_labels(row)[-1]==1
    return dict(status='PASS',valid_binary_csv=True,rejected_cases_n=len(bad_cases),
                normal_and_other_or_correct=True,multi_hot_preserved=True,
                test_accessed=False,gpu_work=False,cpu_seconds=time.monotonic()-begun)


if __name__=='__main__':print(json.dumps(check(),indent=2))
