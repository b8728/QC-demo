from dwave.system import EmbeddingComposite,DWaveSampler,LeapHybridSampler
from neal import SimulatedAnnealingSampler
from pyqubo import Array

#スピンの割り当て (x[0]～x[4]の5つのスピン)
x = Array.create('x',shape=(5),vartype='BINARY')

#定式化 (x[0] + x[1] + x[2] + x[3] + x[4])
H = sum(x[i] for i in range(5))

#QUBOモデル作成
model = H.compile()
qubo,offset = model.to_qubo()

##D-WaveのQPU Samplerを用いてアニーリング（Leap時間を消費します！）
sampler = EmbeddingComposite(DWaveSampler(solver="Advantage_system1.1"))
responses = sampler.sample_qubo(qubo,num_reads=100)

##D-WaveのHybrid Samplerを用いてアニーリング（Leap時間を3秒消費します！）
#※time_limitは3より小さくは設定できません
#sampler_hybrid = LeapHybridSampler()
#responses = sampler_hybrid.sample_qubo(qubo,time_limit=3)

##D-WaveのSimulated Annealing samplerを用いてアニーリング（Leap時間を消費しません）
#sampler_neal = SimulatedAnnealingSampler()
#responses = sampler_neal.sample_qubo(qubo, num_reads=10)

#結果検証
Rlist = []
for s,e,o in responses.data(['sample', 'energy', 'num_occurrences']):
    Rlist.append(s)
    print(s, e, o)

#最良の解の結果を出力
for i in range(5):
    print(f"x[{i}]=",end="")
    print(Rlist[0]['x[%d]' % (i)])

