from dwave.system import EmbeddingComposite,DWaveSampler,LeapHybridSampler
from neal import SimulatedAnnealingSampler
from pyqubo import Array

#スピンの割り当て (x[0]～x[4]の5つのスピン)
x = Array.create('x',shape=(5),vartype='BINARY')

#定式化 
H = sum(x[i] for i in range(5))

#QUBOモデル作成
model = H.compile()
qubo,offset = model.to_qubo()

##D-WaveのSimulated Annealing samplerを用いてアニーリング
sampler_neal = SimulatedAnnealingSampler()
responses = sampler_neal.sample_qubo(qubo, num_reads=10)

#結果検証
Rlist = []
for s,e,o in responses.data(['sample', 'energy', 'num_occurrences']):
    Rlist.append(s)
    print(s, e, o)

#最良の解の結果を出力
for i in range(5):
    print(f"x[{i}]=",end="")
    print(Rlist[0]['x[%d]' % (i)])