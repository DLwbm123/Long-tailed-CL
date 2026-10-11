"""Small trust-boundary check for the official RFMiD CSV parser."""
import csv
import io
import json
import time
from prepare_rfmid_ml import DISEASES, read_labels


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
    return dict(status='PASS',valid_binary_csv=True,rejected_cases_n=len(bad_cases),
                test_accessed=False,gpu_work=False,cpu_seconds=time.monotonic()-begun)


if __name__=='__main__':print(json.dumps(check(),indent=2))
