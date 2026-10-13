# MPC/DSM 同状态最小机制诊断

状态：PASS。四个最终原checkpoint，六个固定诊断头，0 optimizer/0policy/0训练图像/test封存。最终状态诊断不提供新的平均增量或遗忘。
REG_ZERO与ETF_FULL固定同一原γ；IDENTITY为原解析最优头目标（浮点readout原封保留）；ETF_DISC只改变前瞻指定的判别范数缩放。公共分量不影响argmax，但计入原ETF全头尺度。MPC供体权重只支持方向/混淆一致性核对，没有独立真实中心估计。
所有结果为重复development val诊断，不是官方ConCM、训练收益或临床/test证据；不按本结果调参或自动选择头。

|job|head|BA%|ΔBA pp|Acc%|old%|new%|新纠正|新破坏|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|ISIC_base|MPC|56.359697|+0.520833|55.593220|51.472020|71.022727|0|0|
|ISIC_base|REG_ZERO|57.733872|+1.895009|57.288136|53.304254|71.022727|0|0|
|ISIC_base|IDENTITY|55.838864|+0.000000|55.254237|50.777576|71.022727|0|0|
|ISIC_base|ETF_FULL|57.037410|+1.198546|56.610169|51.996849|72.159091|1|0|
|ISIC_base|ETF_DISC|57.037410|+1.198546|56.610169|51.996849|72.159091|1|0|
|ISIC_base|MPC_DSM|57.274152|+1.435288|56.610169|52.691294|71.022727|0|0|
|ISIC_local_transport|MPC|57.365589|+0.236742|56.610169|53.229876|69.772727|0|1|
|ISIC_local_transport|REG_ZERO|57.121224|-0.007622|56.610169|52.941935|69.659091|0|1|
|ISIC_local_transport|IDENTITY|57.128846|+0.000000|56.610169|52.535431|70.909091|0|0|
|ISIC_local_transport|ETF_FULL|57.121224|-0.007622|56.610169|52.941935|69.659091|0|1|
|ISIC_local_transport|ETF_DISC|57.121224|-0.007622|56.610169|52.941935|69.659091|0|1|
|ISIC_local_transport|MPC_DSM|57.357967|+0.229120|56.610169|53.636380|68.522727|0|2|
|HK_base|MPC|61.687702|-0.769238|83.274441|58.861010|91.367970|0|2|
|HK_base|REG_ZERO|62.092157|-0.364783|83.451119|59.334315|91.049498|0|3|
|HK_base|IDENTITY|62.456940|+0.000000|83.922261|59.642054|92.013238|0|0|
|HK_base|ETF_FULL|62.090960|-0.365980|83.568905|59.272343|91.686441|0|1|
|HK_base|ETF_DISC|62.090960|-0.365980|83.568905|59.272343|91.686441|0|1|
|HK_base|MPC_DSM|61.526906|-0.930034|83.097762|58.684900|91.367970|0|2|
|HK_local_transport|MPC|61.894857|-0.575388|83.392226|59.087893|91.367970|0|2|
|HK_local_transport|REG_ZERO|62.059522|-0.410723|83.392226|59.298572|91.049498|0|3|
|HK_local_transport|IDENTITY|62.470245|+0.000000|83.863369|59.656626|92.013238|0|0|
|HK_local_transport|ETF_FULL|62.119990|-0.350255|83.510012|59.304138|91.686441|0|1|
|HK_local_transport|ETF_DISC|62.119990|-0.350255|83.510012|59.304138|91.686441|0|1|
|HK_local_transport|MPC_DSM|61.553231|-0.917014|82.979976|58.744063|91.049498|0|3|

## 正则化/方向/尺度贡献

|job|contrast|BA|Acc|tail|F1|old|new|
|---|---|---:|---:|---:|---:|---:|---:|
|ISIC_base|ETF_FULL−REG_ZERO|-0.696463|-0.677966|-1.961107|-0.682068|-1.307405|+1.136364|
|ISIC_base|ETF_DISC−ETF_FULL|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|
|ISIC_base|MPC_DSM−MPC|+0.914455|+1.016949|+0.609756|+0.789663|+1.219273|+0.000000|
|ISIC_base|MPC_DSM−ETF_FULL|+0.236742|+0.000000|+1.041667|-0.160997|+0.694444|-1.136364|
|ISIC_local_transport|ETF_FULL−REG_ZERO|+0.000000|+0.000000|+0.000000|+0.048151|+0.000000|+0.000000|
|ISIC_local_transport|ETF_DISC−ETF_FULL|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|
|ISIC_local_transport|MPC_DSM−MPC|-0.007622|+0.000000|-0.065920|+0.050755|+0.406504|-1.250000|
|ISIC_local_transport|MPC_DSM−ETF_FULL|+0.236742|+0.000000|+1.041667|-0.178640|+0.694444|-1.136364|
|HK_base|ETF_FULL−REG_ZERO|-0.001196|+0.117786|-0.396825|-0.002488|-0.061971|+0.636943|
|HK_base|ETF_DISC−ETF_FULL|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|
|HK_base|MPC_DSM−MPC|-0.160796|-0.176678|-0.283189|-0.092713|-0.176110|+0.000000|
|HK_base|MPC_DSM−ETF_FULL|-0.564055|-0.471143|-0.694444|-0.387641|-0.587444|-0.318471|
|HK_local_transport|ETF_FULL−REG_ZERO|+0.060468|+0.117786|-0.005411|+0.021534|+0.005566|+0.636943|
|HK_local_transport|ETF_DISC−ETF_FULL|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|+0.000000|
|HK_local_transport|MPC_DSM−MPC|-0.341626|-0.412250|-0.535714|-0.305719|-0.343831|-0.318471|
|HK_local_transport|MPC_DSM−ETF_FULL|-0.566759|-0.530035|-0.808081|-0.358017|-0.560075|-0.636943|

同状态头间新类纠正/破坏：
ISIC_base ETF_FULL−REG_ZERO {"n": 84, "changed": 1, "rescued": 1, "harmed": 0, "wrong_to_different_wrong": 0, "correct_off": 60, "correct_on": 61, "net_correct": 1}
ISIC_base ETF_DISC−ETF_FULL {"n": 84, "changed": 0, "rescued": 0, "harmed": 0, "wrong_to_different_wrong": 0, "correct_off": 61, "correct_on": 61, "net_correct": 0}
ISIC_base MPC_DSM−MPC {"n": 84, "changed": 0, "rescued": 0, "harmed": 0, "wrong_to_different_wrong": 0, "correct_off": 60, "correct_on": 60, "net_correct": 0}
ISIC_base MPC_DSM−ETF_FULL {"n": 84, "changed": 2, "rescued": 0, "harmed": 1, "wrong_to_different_wrong": 1, "correct_off": 61, "correct_on": 60, "net_correct": -1}
ISIC_local_transport ETF_FULL−REG_ZERO {"n": 84, "changed": 0, "rescued": 0, "harmed": 0, "wrong_to_different_wrong": 0, "correct_off": 59, "correct_on": 59, "net_correct": 0}
ISIC_local_transport ETF_DISC−ETF_FULL {"n": 84, "changed": 0, "rescued": 0, "harmed": 0, "wrong_to_different_wrong": 0, "correct_off": 59, "correct_on": 59, "net_correct": 0}
ISIC_local_transport MPC_DSM−MPC {"n": 84, "changed": 1, "rescued": 0, "harmed": 1, "wrong_to_different_wrong": 0, "correct_off": 59, "correct_on": 58, "net_correct": -1}
ISIC_local_transport MPC_DSM−ETF_FULL {"n": 84, "changed": 2, "rescued": 0, "harmed": 1, "wrong_to_different_wrong": 1, "correct_off": 59, "correct_on": 58, "net_correct": -1}
HK_base ETF_FULL−REG_ZERO {"n": 310, "changed": 2, "rescued": 2, "harmed": 0, "wrong_to_different_wrong": 0, "correct_off": 282, "correct_on": 284, "net_correct": 2}
HK_base ETF_DISC−ETF_FULL {"n": 310, "changed": 0, "rescued": 0, "harmed": 0, "wrong_to_different_wrong": 0, "correct_off": 284, "correct_on": 284, "net_correct": 0}
HK_base MPC_DSM−MPC {"n": 310, "changed": 0, "rescued": 0, "harmed": 0, "wrong_to_different_wrong": 0, "correct_off": 283, "correct_on": 283, "net_correct": 0}
HK_base MPC_DSM−ETF_FULL {"n": 310, "changed": 1, "rescued": 0, "harmed": 1, "wrong_to_different_wrong": 0, "correct_off": 284, "correct_on": 283, "net_correct": -1}
HK_local_transport ETF_FULL−REG_ZERO {"n": 310, "changed": 2, "rescued": 2, "harmed": 0, "wrong_to_different_wrong": 0, "correct_off": 282, "correct_on": 284, "net_correct": 2}
HK_local_transport ETF_DISC−ETF_FULL {"n": 310, "changed": 0, "rescued": 0, "harmed": 0, "wrong_to_different_wrong": 0, "correct_off": 284, "correct_on": 284, "net_correct": 0}
HK_local_transport MPC_DSM−MPC {"n": 310, "changed": 1, "rescued": 0, "harmed": 1, "wrong_to_different_wrong": 0, "correct_off": 283, "correct_on": 282, "net_correct": -1}
HK_local_transport MPC_DSM−ETF_FULL {"n": 310, "changed": 2, "rescued": 0, "harmed": 2, "wrong_to_different_wrong": 0, "correct_off": 284, "correct_on": 282, "net_correct": -2}

## 全头、判别及公共范数

|job|head|full|disc|common|公共能量%|γ|
|---|---|---:|---:|---:|---:|---:|
|ISIC_base|MPC|10.880864012|10.866288782|0.563000669|0.267726|0|
|ISIC_base|REG_ZERO|10.033038280|10.017745480|0.553744176|0.304616|0.00016472444|
|ISIC_base|IDENTITY|10.720514366|10.705719648|0.563023183|0.275817|0.00016472444|
|ISIC_base|ETF_FULL|10.306899483|10.292013620|0.553744176|0.288644|0.00016472444|
|ISIC_base|ETF_DISC|10.306520177|10.291633764|0.553744176|0.288665|0.00016472444|
|ISIC_base|MPC_DSM|10.454532751|10.439857869|0.553735233|0.280540|0.000164450744|

### ISIC_base 原头与供体核对

{"full_norm": 10.720514366376133, "discriminative_norm": 10.70571964769342, "common_norm": 0.5630231830645843, "common_energy_fraction": 0.0027581717703909488, "orthogonal_energy_error": 1.2364852873556758e-16}

|class|n|主要供体(权重/方向cos)|有效供体数|错误类供体质量|错向权重重合|
|---|---:|---|---:|---:|---:|
|4|458|2 (0.2797/0.7412); 5 (0.2120/0.8296); 3 (0.1892/0.9218)|5.1436|0.746897|0.195938|
|7|27|3 (0.1898/0.9379); 5 (0.1882/0.8557); 0 (0.1863/0.6610)|6.4405|0.914779|0.151486|
|5|256|3 (0.2225/0.7879); 2 (0.2110/0.6931); 4 (0.1806/0.5262)|5.8878|0.703425|0.200229|
|1|3263|3 (0.2383/0.8693); 2 (0.1853/0.8289); 5 (0.1709/0.8705)|5.7348|0.965824|0.183780|
|2|2835|4 (0.2321/0.2903); 5 (0.2055/0.7519); 3 (0.1954/0.9029)|5.7022|0.885552|0.194873|
|6|44|0 (0.2161/0.8233); 3 (0.1690/0.9694); 2 (0.1603/0.9079)|6.4678|0.376466|0.178933|
|0|10529|3 (0.2418/0.9188); 1 (0.1978/0.7790); 5 (0.1374/0.9416)|6.1374|0.906209|0.166608|
|3|1306|5 (0.1995/0.6072); 1 (0.1912/0.4519); 2 (0.1799/0.7718)|6.1382|0.753013|0.158443|
|ISIC_local_transport|MPC|11.154261658|11.139824965|0.567320804|0.258688|0|
|ISIC_local_transport|REG_ZERO|10.159969989|10.144781273|0.555340521|0.298768|0.000164661242|
|ISIC_local_transport|IDENTITY|10.992593416|10.977944606|0.567311417|0.266344|0.000164661242|
|ISIC_local_transport|ETF_FULL|10.445329658|10.430556485|0.555340521|0.282667|0.000164661242|
|ISIC_local_transport|ETF_DISC|10.444948011|10.430174297|0.555340521|0.282687|0.000164661242|
|ISIC_local_transport|MPC_DSM|10.595041003|10.580476089|0.555355376|0.274749|0.000164386219|

### ISIC_local_transport 原头与供体核对

{"full_norm": 10.9925934162572, "discriminative_norm": 10.977944605961834, "common_norm": 0.5673114167495728, "common_energy_fraction": 0.0026634387692165335, "orthogonal_energy_error": 1.1760339777590983e-16}

|class|n|主要供体(权重/方向cos)|有效供体数|错误类供体质量|错向权重重合|
|---|---:|---|---:|---:|---:|
|4|458|2 (0.2792/0.7415); 5 (0.2118/0.8299); 3 (0.1902/0.9237)|5.1443|0.910176|0.192518|
|7|27|3 (0.1907/0.9390); 5 (0.1883/0.8563); 0 (0.1844/0.6579)|6.4454|0.914597|0.151318|
|5|256|3 (0.2235/0.7901); 2 (0.2108/0.6943); 4 (0.1806/0.5271)|5.8814|0.704230|0.199494|
|1|3263|3 (0.2389/0.8721); 2 (0.1855/0.8298); 5 (0.1710/0.8712)|5.7333|0.965681|0.182204|
|2|2835|4 (0.2318/0.2911); 5 (0.2053/0.7528); 3 (0.1967/0.9039)|5.6986|0.885325|0.193568|
|6|44|0 (0.2144/0.8207); 3 (0.1697/0.9701); 2 (0.1606/0.9081)|6.4783|0.374979|0.178511|
|0|10529|3 (0.2414/0.9222); 1 (0.1978/0.7812); 5 (0.1374/0.9420)|6.1412|0.827976|0.177683|
|3|1306|5 (0.2000/0.5997); 1 (0.1911/0.4434); 2 (0.1806/0.7685)|6.1385|0.752356|0.158296|
|HK_base|MPC|12.098543055|12.094441567|0.315003520|0.067790|0|
|HK_base|REG_ZERO|11.424251481|11.420016871|0.311025054|0.074120|0.000164439787|
|HK_base|IDENTITY|12.172105175|12.168017302|0.315435144|0.067157|0.000164439787|
|HK_base|ETF_FULL|11.886527649|11.882457783|0.311025054|0.068467|0.000164439787|
|HK_base|ETF_DISC|11.886371095|11.882301176|0.311025054|0.068469|0.000164439787|
|HK_base|MPC_DSM|11.815471827|11.811388400|0.310609969|0.069108|0.000163727996|

### HK_base 原头与供体核对

{"full_norm": 12.172105174974178, "discriminative_norm": 12.168017301951696, "common_norm": 0.3154351439472248, "common_energy_fraction": 0.0006715660979289395, "orthogonal_energy_error": 0.0}

|class|n|主要供体(权重/方向cos)|有效供体数|错误类供体质量|错向权重重合|
|---|---:|---|---:|---:|---:|
|4|626|3 (0.4977/0.7378); 13 (0.0590/0.8899); 14 (0.0557/0.7538)|3.7242|0.497670|0.497670|
|5|3|14 (0.2939/0.8273); 22 (0.0875/0.8478); 20 (0.0834/0.9400)|8.0445|0.293874|0.293874|
|6|6|20 (0.1588/0.9348); 18 (0.1434/0.9412); 22 (0.0920/0.8266)|10.9447|0.158763|0.158763|
|22|85|20 (0.2243/0.8728); 18 (0.1602/0.8879); 21 (0.1219/0.5474)|8.7986|0.427411|0.111867|
|20|288|18 (0.1853/0.5441); 22 (0.1477/0.0431); 21 (0.1018/-0.0845)|10.1941|0.900535|0.116233|
|1|403|8 (0.1176/0.6470); 20 (0.1153/0.8881); 2 (0.1153/0.5432)|11.5352|0.115254|0.115254|
|2|736|7 (0.2019/0.4287); 1 (0.1476/0.7827); 20 (0.0913/0.9093)|9.9131|0.201850|0.201850|
|15|488|9 (0.1815/0.8409); 11 (0.1658/0.9083); 10 (0.1309/0.9087)|9.3173|0.066842|0.066842|
|14|252|18 (0.1289/0.9186); 20 (0.1276/0.9150); 13 (0.1026/0.8217)|12.7671|0.156023|0.078011|
|12|167|11 (0.2693/0.9020); 16 (0.2032/0.8719); 0 (0.1909/0.7795)|5.4233|0.834465|0.218420|
|10|595|11 (0.3007/0.8212); 16 (0.1899/0.8226); 0 (0.1506/0.7956)|5.3907|0.906144|0.207206|
|11|261|10 (0.2627/0.4002); 16 (0.2034/0.5578); 12 (0.1980/0.5024)|5.3812|0.918065|0.215955|
|19|6|20 (0.1822/0.8933); 18 (0.1640/0.8995); 22 (0.0994/0.7466)|9.8656|0.251875|0.125938|
|16|34|11 (0.2673/0.7084); 10 (0.2180/0.6323); 12 (0.1964/0.4684)|5.2025|0.657767|0.200518|
|7|83|2 (0.2740/0.6808); 1 (0.1866/0.8003); 13 (0.0764/0.8544)|7.2785|0.000000|无错误|
|0|25|11 (0.2541/0.8285); 12 (0.2141/0.4562); 10 (0.2006/0.7430)|5.1316|0.454668|0.227334|
|17|23|18 (0.1901/0.8186); 20 (0.1528/0.8551); 13 (0.1069/0.6565)|9.8719|0.210106|0.105053|
|9|635|10 (0.2277/0.8317); 11 (0.1887/0.8864); 16 (0.1075/0.9032)|7.9994|0.020814|0.010407|
|18|127|20 (0.1942/0.5199); 22 (0.1106/0.4333); 17 (0.1045/0.1139)|10.2399|0.715095|0.106618|
|13|662|18 (0.1384/0.8761); 20 (0.1224/0.8923); 8 (0.1112/0.6011)|12.4493|0.296318|0.049193|
|21|18|20 (0.2160/0.8576); 22 (0.1704/0.4720); 18 (0.1523/0.8696)|8.4078|0.386423|0.181799|
|8|654|18 (0.1494/0.8943); 13 (0.1349/0.7601); 20 (0.1207/0.9105)|10.8173|0.078538|0.078538|
|3|644|4 (0.3592/0.1649); 13 (0.0932/0.7766); 8 (0.0739/0.7370)|6.0821|0.496403|0.332784|
|HK_local_transport|MPC|12.391925262|12.387826513|0.318694190|0.066141|0|
|HK_local_transport|REG_ZERO|11.586562291|11.582334880|0.312960792|0.072958|0.000163975147|
|HK_local_transport|IDENTITY|12.455522109|12.451432653|0.319148710|0.065654|0.000163975147|
|HK_local_transport|ETF_FULL|12.067096710|12.063037700|0.312960792|0.067263|0.000163975147|
|HK_local_transport|ETF_DISC|12.066937590|12.062878527|0.312960792|0.067264|0.000163975147|
|HK_local_transport|MPC_DSM|12.002912743|11.998843073|0.312536764|0.067800|0.000163263404|

### HK_local_transport 原头与供体核对

{"full_norm": 12.455522109070078, "discriminative_norm": 12.451432652918717, "common_norm": 0.3191487104198252, "common_energy_fraction": 0.0006565416978444287, "orthogonal_energy_error": 3.664007186985481e-16}

|class|n|主要供体(权重/方向cos)|有效供体数|错误类供体质量|错向权重重合|
|---|---:|---|---:|---:|---:|
|4|626|3 (0.4973/0.7375); 13 (0.0591/0.8900); 14 (0.0557/0.7538)|3.7284|0.497336|0.497336|
|5|3|14 (0.2936/0.8272); 22 (0.0875/0.8478); 20 (0.0834/0.9400)|8.0520|0.293635|0.293635|
|6|6|20 (0.1587/0.9348); 18 (0.1434/0.9411); 22 (0.0920/0.8266)|10.9464|0.158704|0.158704|
|22|85|20 (0.2242/0.8729); 18 (0.1601/0.8877); 21 (0.1218/0.5475)|8.8034|0.427240|0.111805|
|20|288|18 (0.1852/0.5434); 22 (0.1477/0.0428); 21 (0.1017/-0.0849)|10.1976|0.900486|0.116196|
|1|403|8 (0.1175/0.6473); 20 (0.1153/0.8881); 2 (0.1153/0.5432)|11.5378|0.115302|0.115302|
|2|736|7 (0.2018/0.4284); 1 (0.1477/0.7826); 20 (0.0914/0.9094)|9.9155|0.201758|0.201758|
|15|488|9 (0.1815/0.8409); 11 (0.1658/0.9083); 10 (0.1309/0.9087)|9.3181|0.066784|0.066784|
|14|252|18 (0.1288/0.9186); 20 (0.1276/0.9151); 13 (0.1026/0.8218)|12.7680|0.155909|0.077955|
|12|167|11 (0.2692/0.9020); 16 (0.2032/0.8716); 0 (0.1909/0.7792)|5.4249|0.834357|0.218370|
|10|595|11 (0.3007/0.8210); 16 (0.1898/0.8226); 0 (0.1505/0.7957)|5.3918|0.906039|0.207110|
|11|261|10 (0.2627/0.3997); 16 (0.2033/0.5579); 12 (0.1980/0.5029)|5.3823|0.917981|0.216352|
|19|6|20 (0.1821/0.8933); 18 (0.1640/0.8994); 22 (0.0994/0.7467)|9.8694|0.251809|0.125904|
|16|34|11 (0.2673/0.7086); 10 (0.2179/0.6325); 12 (0.1965/0.4679)|5.2036|0.657589|0.200478|
|7|83|2 (0.2739/0.6810); 1 (0.1865/0.8003); 13 (0.0764/0.8546)|7.2797|0.000000|无错误|
|0|25|11 (0.2540/0.8283); 12 (0.2142/0.4558); 10 (0.2005/0.7430)|5.1324|0.454516|0.227258|
|17|23|18 (0.1900/0.8184); 20 (0.1528/0.8551); 13 (0.1069/0.6567)|9.8747|0.210009|0.105005|
|9|635|10 (0.2277/0.8316); 11 (0.1886/0.8862); 16 (0.1075/0.9031)|8.0022|0.020818|0.010409|
|18|127|20 (0.1942/0.5205); 22 (0.1106/0.4335); 17 (0.1045/0.1147)|10.2407|0.715160|0.106616|
|13|662|18 (0.1383/0.8760); 20 (0.1224/0.8923); 8 (0.1112/0.6009)|12.4520|0.296309|0.049191|
|21|18|20 (0.2160/0.8577); 22 (0.1703/0.4722); 18 (0.1523/0.8693)|8.4120|0.386209|0.181682|
|8|654|18 (0.1494/0.8942); 13 (0.1349/0.7601); 20 (0.1207/0.9106)|10.8163|0.078566|0.078566|
|3|644|4 (0.3590/0.1651); 13 (0.0933/0.7768); 8 (0.0739/0.7372)|6.0867|0.496323|0.332603|

## 全部类纠正和破坏

|job|head|class|val n|OFF正确|ON正确|纠正|破坏|错转错|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|ISIC_base|MPC|4|41|17|17|0|0|0|
|ISIC_base|MPC|7|24|11|12|1|0|0|
|ISIC_base|MPC|5|37|11|11|0|0|0|
|ISIC_base|MPC|1|37|19|19|0|0|2|
|ISIC_base|MPC|2|46|22|22|0|0|0|
|ISIC_base|MPC|6|26|23|23|0|0|0|
|ISIC_base|MPC|0|44|35|35|0|0|1|
|ISIC_base|MPC|3|40|25|25|0|0|1|
|ISIC_base|REG_ZERO|4|41|17|19|2|0|1|
|ISIC_base|REG_ZERO|7|24|11|11|0|0|1|
|ISIC_base|REG_ZERO|5|37|11|13|2|0|0|
|ISIC_base|REG_ZERO|1|37|19|20|1|0|1|
|ISIC_base|REG_ZERO|2|46|22|23|1|0|0|
|ISIC_base|REG_ZERO|6|26|23|23|0|0|0|
|ISIC_base|REG_ZERO|0|44|35|35|0|0|1|
|ISIC_base|REG_ZERO|3|40|25|25|0|0|0|
|ISIC_base|IDENTITY|4|41|17|17|0|0|0|
|ISIC_base|IDENTITY|7|24|11|11|0|0|0|
|ISIC_base|IDENTITY|5|37|11|11|0|0|0|
|ISIC_base|IDENTITY|1|37|19|19|0|0|0|
|ISIC_base|IDENTITY|2|46|22|22|0|0|0|
|ISIC_base|IDENTITY|6|26|23|23|0|0|0|
|ISIC_base|IDENTITY|0|44|35|35|0|0|0|
|ISIC_base|IDENTITY|3|40|25|25|0|0|0|
|ISIC_base|ETF_FULL|4|41|17|18|2|1|1|
|ISIC_base|ETF_FULL|7|24|11|11|0|0|1|
|ISIC_base|ETF_FULL|5|37|11|11|0|0|0|
|ISIC_base|ETF_FULL|1|37|19|20|1|0|1|
|ISIC_base|ETF_FULL|2|46|22|23|1|0|0|
|ISIC_base|ETF_FULL|6|26|23|23|0|0|0|
|ISIC_base|ETF_FULL|0|44|35|36|1|0|0|
|ISIC_base|ETF_FULL|3|40|25|25|0|0|0|
|ISIC_base|ETF_DISC|4|41|17|18|2|1|1|
|ISIC_base|ETF_DISC|7|24|11|11|0|0|1|
|ISIC_base|ETF_DISC|5|37|11|11|0|0|0|
|ISIC_base|ETF_DISC|1|37|19|20|1|0|1|
|ISIC_base|ETF_DISC|2|46|22|23|1|0|0|
|ISIC_base|ETF_DISC|6|26|23|23|0|0|0|
|ISIC_base|ETF_DISC|0|44|35|36|1|0|0|
|ISIC_base|ETF_DISC|3|40|25|25|0|0|0|
|ISIC_base|MPC_DSM|4|41|17|18|2|1|1|
|ISIC_base|MPC_DSM|7|24|11|12|1|0|1|
|ISIC_base|MPC_DSM|5|37|11|11|0|0|1|
|ISIC_base|MPC_DSM|1|37|19|20|1|0|2|
|ISIC_base|MPC_DSM|2|46|22|23|1|0|0|
|ISIC_base|MPC_DSM|6|26|23|23|0|0|0|
|ISIC_base|MPC_DSM|0|44|35|35|0|0|1|
|ISIC_base|MPC_DSM|3|40|25|25|0|0|1|
|ISIC_local_transport|MPC|4|41|18|18|0|0|1|
|ISIC_local_transport|MPC|7|24|11|12|1|0|1|
|ISIC_local_transport|MPC|5|37|15|15|0|0|1|
|ISIC_local_transport|MPC|1|37|18|18|0|0|1|
|ISIC_local_transport|MPC|2|46|22|22|0|0|0|
|ISIC_local_transport|MPC|6|26|23|23|0|0|0|
|ISIC_local_transport|MPC|0|44|36|35|0|1|0|
|ISIC_local_transport|MPC|3|40|24|24|0|0|1|
|ISIC_local_transport|REG_ZERO|4|41|18|19|1|0|0|
|ISIC_local_transport|REG_ZERO|7|24|11|11|0|0|0|
|ISIC_local_transport|REG_ZERO|5|37|15|14|0|1|2|
|ISIC_local_transport|REG_ZERO|1|37|18|19|1|0|1|
|ISIC_local_transport|REG_ZERO|2|46|22|22|1|1|0|
|ISIC_local_transport|REG_ZERO|6|26|23|23|0|0|0|
|ISIC_local_transport|REG_ZERO|0|44|36|36|0|0|0|
|ISIC_local_transport|REG_ZERO|3|40|24|23|0|1|0|
|ISIC_local_transport|IDENTITY|4|41|18|18|0|0|0|
|ISIC_local_transport|IDENTITY|7|24|11|11|0|0|0|
|ISIC_local_transport|IDENTITY|5|37|15|15|0|0|0|
|ISIC_local_transport|IDENTITY|1|37|18|18|0|0|0|
|ISIC_local_transport|IDENTITY|2|46|22|22|0|0|0|
|ISIC_local_transport|IDENTITY|6|26|23|23|0|0|0|
|ISIC_local_transport|IDENTITY|0|44|36|36|0|0|0|
|ISIC_local_transport|IDENTITY|3|40|24|24|0|0|0|
|ISIC_local_transport|ETF_FULL|4|41|18|19|1|0|1|
|ISIC_local_transport|ETF_FULL|7|24|11|11|0|0|0|
|ISIC_local_transport|ETF_FULL|5|37|15|14|0|1|2|
|ISIC_local_transport|ETF_FULL|1|37|18|19|1|0|1|
|ISIC_local_transport|ETF_FULL|2|46|22|22|1|1|0|
|ISIC_local_transport|ETF_FULL|6|26|23|23|0|0|0|
|ISIC_local_transport|ETF_FULL|0|44|36|36|0|0|0|
|ISIC_local_transport|ETF_FULL|3|40|24|23|0|1|0|
|ISIC_local_transport|ETF_DISC|4|41|18|19|1|0|1|
|ISIC_local_transport|ETF_DISC|7|24|11|11|0|0|0|
|ISIC_local_transport|ETF_DISC|5|37|15|14|0|1|2|
|ISIC_local_transport|ETF_DISC|1|37|18|19|1|0|1|
|ISIC_local_transport|ETF_DISC|2|46|22|22|1|1|0|
|ISIC_local_transport|ETF_DISC|6|26|23|23|0|0|0|
|ISIC_local_transport|ETF_DISC|0|44|36|36|0|0|0|
|ISIC_local_transport|ETF_DISC|3|40|24|23|0|1|0|
|ISIC_local_transport|MPC_DSM|4|41|18|19|1|0|1|
|ISIC_local_transport|MPC_DSM|7|24|11|12|1|0|1|
|ISIC_local_transport|MPC_DSM|5|37|15|14|0|1|2|
|ISIC_local_transport|MPC_DSM|1|37|18|19|1|0|2|
|ISIC_local_transport|MPC_DSM|2|46|22|22|1|1|0|
|ISIC_local_transport|MPC_DSM|6|26|23|23|0|0|0|
|ISIC_local_transport|MPC_DSM|0|44|36|35|0|1|0|
|ISIC_local_transport|MPC_DSM|3|40|24|23|0|1|1|
|HK_base|MPC|4|165|142|143|1|0|0|
|HK_base|MPC|5|1|0|0|0|0|0|
|HK_base|MPC|6|1|0|0|0|0|0|
|HK_base|MPC|22|21|15|15|0|0|0|
|HK_base|MPC|20|66|16|15|0|1|3|
|HK_base|MPC|1|113|107|107|0|0|0|
|HK_base|MPC|2|182|154|152|0|2|0|
|HK_base|MPC|15|123|122|122|0|0|0|
|HK_base|MPC|14|60|58|53|0|5|0|
|HK_base|MPC|12|41|33|33|0|0|0|
|HK_base|MPC|10|150|115|115|0|0|0|
|HK_base|MPC|11|61|17|17|0|0|0|
|HK_base|MPC|19|2|0|0|0|0|0|
|HK_base|MPC|16|8|3|3|0|0|0|
|HK_base|MPC|7|21|21|21|0|0|0|
|HK_base|MPC|0|7|5|5|0|0|0|
|HK_base|MPC|17|5|3|3|0|0|0|
|HK_base|MPC|9|164|162|162|0|0|0|
|HK_base|MPC|18|33|16|14|0|2|1|
|HK_base|MPC|13|160|151|151|0|0|0|
|HK_base|MPC|21|4|0|0|0|0|0|
|HK_base|MPC|8|153|150|149|0|1|0|
|HK_base|MPC|3|157|135|134|0|1|0|
|HK_base|REG_ZERO|4|165|142|142|0|0|0|
|HK_base|REG_ZERO|5|1|0|0|0|0|0|
|HK_base|REG_ZERO|6|1|0|0|0|0|0|
|HK_base|REG_ZERO|22|21|15|15|0|0|0|
|HK_base|REG_ZERO|20|66|16|16|0|0|0|
|HK_base|REG_ZERO|1|113|107|107|0|0|0|
|HK_base|REG_ZERO|2|182|154|152|0|2|0|
|HK_base|REG_ZERO|15|123|122|122|0|0|0|
|HK_base|REG_ZERO|14|60|58|57|0|1|0|
|HK_base|REG_ZERO|12|41|33|33|0|0|0|
|HK_base|REG_ZERO|10|150|115|114|0|1|0|
|HK_base|REG_ZERO|11|61|17|17|1|1|0|
|HK_base|REG_ZERO|19|2|0|0|0|0|0|
|HK_base|REG_ZERO|16|8|3|3|0|0|0|
|HK_base|REG_ZERO|7|21|21|21|0|0|0|
|HK_base|REG_ZERO|0|7|5|5|0|0|0|
|HK_base|REG_ZERO|17|5|3|3|0|0|0|
|HK_base|REG_ZERO|9|164|162|162|0|0|0|
|HK_base|REG_ZERO|18|33|16|15|0|1|1|
|HK_base|REG_ZERO|13|160|151|151|0|0|1|
|HK_base|REG_ZERO|21|4|0|0|0|0|0|
|HK_base|REG_ZERO|8|153|150|149|0|1|0|
|HK_base|REG_ZERO|3|157|135|133|0|2|0|
|HK_base|IDENTITY|4|165|142|142|0|0|0|
|HK_base|IDENTITY|5|1|0|0|0|0|0|
|HK_base|IDENTITY|6|1|0|0|0|0|0|
|HK_base|IDENTITY|22|21|15|15|0|0|0|
|HK_base|IDENTITY|20|66|16|16|0|0|0|
|HK_base|IDENTITY|1|113|107|107|0|0|0|
|HK_base|IDENTITY|2|182|154|154|0|0|0|
|HK_base|IDENTITY|15|123|122|122|0|0|0|
|HK_base|IDENTITY|14|60|58|58|0|0|0|
|HK_base|IDENTITY|12|41|33|33|0|0|0|
|HK_base|IDENTITY|10|150|115|115|0|0|0|
|HK_base|IDENTITY|11|61|17|17|0|0|0|
|HK_base|IDENTITY|19|2|0|0|0|0|0|
|HK_base|IDENTITY|16|8|3|3|0|0|0|
|HK_base|IDENTITY|7|21|21|21|0|0|0|
|HK_base|IDENTITY|0|7|5|5|0|0|0|
|HK_base|IDENTITY|17|5|3|3|0|0|0|
|HK_base|IDENTITY|9|164|162|162|0|0|0|
|HK_base|IDENTITY|18|33|16|16|0|0|0|
|HK_base|IDENTITY|13|160|151|151|0|0|0|
|HK_base|IDENTITY|21|4|0|0|0|0|0|
|HK_base|IDENTITY|8|153|150|150|0|0|0|
|HK_base|IDENTITY|3|157|135|135|0|0|0|
|HK_base|ETF_FULL|4|165|142|142|0|0|0|
|HK_base|ETF_FULL|5|1|0|0|0|0|0|
|HK_base|ETF_FULL|6|1|0|0|0|0|0|
|HK_base|ETF_FULL|22|21|15|14|0|1|0|
|HK_base|ETF_FULL|20|66|16|17|1|0|0|
|HK_base|ETF_FULL|1|113|107|107|0|0|0|
|HK_base|ETF_FULL|2|182|154|152|0|2|0|
|HK_base|ETF_FULL|15|123|122|122|0|0|0|
|HK_base|ETF_FULL|14|60|58|57|0|1|0|
|HK_base|ETF_FULL|12|41|33|33|0|0|0|
|HK_base|ETF_FULL|10|150|115|112|0|3|0|
|HK_base|ETF_FULL|11|61|17|19|2|0|0|
|HK_base|ETF_FULL|19|2|0|0|0|0|0|
|HK_base|ETF_FULL|16|8|3|3|0|0|0|
|HK_base|ETF_FULL|7|21|21|21|0|0|0|
|HK_base|ETF_FULL|0|7|5|5|0|0|0|
|HK_base|ETF_FULL|17|5|3|3|0|0|0|
|HK_base|ETF_FULL|9|164|162|162|0|0|0|
|HK_base|ETF_FULL|18|33|16|15|0|1|0|
|HK_base|ETF_FULL|13|160|151|151|0|0|1|
|HK_base|ETF_FULL|21|4|0|0|0|0|0|
|HK_base|ETF_FULL|8|153|150|149|0|1|0|
|HK_base|ETF_FULL|3|157|135|135|0|0|0|
|HK_base|ETF_DISC|4|165|142|142|0|0|0|
|HK_base|ETF_DISC|5|1|0|0|0|0|0|
|HK_base|ETF_DISC|6|1|0|0|0|0|0|
|HK_base|ETF_DISC|22|21|15|14|0|1|0|
|HK_base|ETF_DISC|20|66|16|17|1|0|0|
|HK_base|ETF_DISC|1|113|107|107|0|0|0|
|HK_base|ETF_DISC|2|182|154|152|0|2|0|
|HK_base|ETF_DISC|15|123|122|122|0|0|0|
|HK_base|ETF_DISC|14|60|58|57|0|1|0|
|HK_base|ETF_DISC|12|41|33|33|0|0|0|
|HK_base|ETF_DISC|10|150|115|112|0|3|0|
|HK_base|ETF_DISC|11|61|17|19|2|0|0|
|HK_base|ETF_DISC|19|2|0|0|0|0|0|
|HK_base|ETF_DISC|16|8|3|3|0|0|0|
|HK_base|ETF_DISC|7|21|21|21|0|0|0|
|HK_base|ETF_DISC|0|7|5|5|0|0|0|
|HK_base|ETF_DISC|17|5|3|3|0|0|0|
|HK_base|ETF_DISC|9|164|162|162|0|0|0|
|HK_base|ETF_DISC|18|33|16|15|0|1|0|
|HK_base|ETF_DISC|13|160|151|151|0|0|1|
|HK_base|ETF_DISC|21|4|0|0|0|0|0|
|HK_base|ETF_DISC|8|153|150|149|0|1|0|
|HK_base|ETF_DISC|3|157|135|135|0|0|0|
|HK_base|MPC_DSM|4|165|142|142|0|0|0|
|HK_base|MPC_DSM|5|1|0|0|0|0|0|
|HK_base|MPC_DSM|6|1|0|0|0|0|0|
|HK_base|MPC_DSM|22|21|15|14|0|1|0|
|HK_base|MPC_DSM|20|66|16|15|1|2|3|
|HK_base|MPC_DSM|1|113|107|107|0|0|0|
|HK_base|MPC_DSM|2|182|154|152|0|2|0|
|HK_base|MPC_DSM|15|123|122|122|0|0|0|
|HK_base|MPC_DSM|14|60|58|52|0|6|0|
|HK_base|MPC_DSM|12|41|33|33|0|0|0|
|HK_base|MPC_DSM|10|150|115|113|0|2|0|
|HK_base|MPC_DSM|11|61|17|18|1|0|0|
|HK_base|MPC_DSM|19|2|0|0|0|0|0|
|HK_base|MPC_DSM|16|8|3|3|0|0|0|
|HK_base|MPC_DSM|7|21|21|21|0|0|0|
|HK_base|MPC_DSM|0|7|5|5|0|0|0|
|HK_base|MPC_DSM|17|5|3|3|0|0|0|
|HK_base|MPC_DSM|9|164|162|162|0|0|0|
|HK_base|MPC_DSM|18|33|16|15|0|1|0|
|HK_base|MPC_DSM|13|160|151|151|0|0|0|
|HK_base|MPC_DSM|21|4|0|0|0|0|0|
|HK_base|MPC_DSM|8|153|150|149|0|1|0|
|HK_base|MPC_DSM|3|157|135|134|0|1|0|
|HK_local_transport|MPC|4|165|142|142|1|1|0|
|HK_local_transport|MPC|5|1|0|0|0|0|0|
|HK_local_transport|MPC|6|1|0|0|0|0|0|
|HK_local_transport|MPC|22|21|15|15|0|0|0|
|HK_local_transport|MPC|20|66|16|14|0|2|1|
|HK_local_transport|MPC|1|113|107|107|0|0|0|
|HK_local_transport|MPC|2|182|154|153|0|1|0|
|HK_local_transport|MPC|15|123|122|122|0|0|0|
|HK_local_transport|MPC|14|60|58|54|0|4|0|
|HK_local_transport|MPC|12|41|33|33|0|0|0|
|HK_local_transport|MPC|10|150|113|115|2|0|0|
|HK_local_transport|MPC|11|61|18|18|0|0|1|
|HK_local_transport|MPC|19|2|0|0|0|0|0|
|HK_local_transport|MPC|16|8|3|3|0|0|0|
|HK_local_transport|MPC|7|21|21|21|0|0|0|
|HK_local_transport|MPC|0|7|5|5|0|0|0|
|HK_local_transport|MPC|17|5|3|3|0|0|0|
|HK_local_transport|MPC|9|164|162|162|0|0|0|
|HK_local_transport|MPC|18|33|16|15|0|1|0|
|HK_local_transport|MPC|13|160|151|151|0|0|0|
|HK_local_transport|MPC|21|4|0|0|0|0|0|
|HK_local_transport|MPC|8|153|150|149|0|1|0|
|HK_local_transport|MPC|3|157|135|134|0|1|0|
|HK_local_transport|REG_ZERO|4|165|142|141|0|1|0|
|HK_local_transport|REG_ZERO|5|1|0|0|0|0|0|
|HK_local_transport|REG_ZERO|6|1|0|0|0|0|0|
|HK_local_transport|REG_ZERO|22|21|15|15|0|0|0|
|HK_local_transport|REG_ZERO|20|66|16|16|0|0|0|
|HK_local_transport|REG_ZERO|1|113|107|107|0|0|0|
|HK_local_transport|REG_ZERO|2|182|154|153|0|1|0|
|HK_local_transport|REG_ZERO|15|123|122|122|0|0|0|
|HK_local_transport|REG_ZERO|14|60|58|56|0|2|0|
|HK_local_transport|REG_ZERO|12|41|33|33|0|0|0|
|HK_local_transport|REG_ZERO|10|150|113|113|0|0|0|
|HK_local_transport|REG_ZERO|11|61|18|18|0|0|0|
|HK_local_transport|REG_ZERO|19|2|0|0|0|0|0|
|HK_local_transport|REG_ZERO|16|8|3|3|0|0|0|
|HK_local_transport|REG_ZERO|7|21|21|21|0|0|0|
|HK_local_transport|REG_ZERO|0|7|5|5|0|0|0|
|HK_local_transport|REG_ZERO|17|5|3|3|0|0|0|
|HK_local_transport|REG_ZERO|9|164|162|162|0|0|0|
|HK_local_transport|REG_ZERO|18|33|16|15|0|1|0|
|HK_local_transport|REG_ZERO|13|160|151|151|0|0|0|
|HK_local_transport|REG_ZERO|21|4|0|0|0|0|0|
|HK_local_transport|REG_ZERO|8|153|150|149|0|1|0|
|HK_local_transport|REG_ZERO|3|157|135|133|0|2|0|
|HK_local_transport|IDENTITY|4|165|142|142|0|0|0|
|HK_local_transport|IDENTITY|5|1|0|0|0|0|0|
|HK_local_transport|IDENTITY|6|1|0|0|0|0|0|
|HK_local_transport|IDENTITY|22|21|15|15|0|0|0|
|HK_local_transport|IDENTITY|20|66|16|16|0|0|0|
|HK_local_transport|IDENTITY|1|113|107|107|0|0|0|
|HK_local_transport|IDENTITY|2|182|154|154|0|0|0|
|HK_local_transport|IDENTITY|15|123|122|122|0|0|0|
|HK_local_transport|IDENTITY|14|60|58|58|0|0|0|
|HK_local_transport|IDENTITY|12|41|33|33|0|0|0|
|HK_local_transport|IDENTITY|10|150|113|113|0|0|0|
|HK_local_transport|IDENTITY|11|61|18|18|0|0|0|
|HK_local_transport|IDENTITY|19|2|0|0|0|0|0|
|HK_local_transport|IDENTITY|16|8|3|3|0|0|0|
|HK_local_transport|IDENTITY|7|21|21|21|0|0|0|
|HK_local_transport|IDENTITY|0|7|5|5|0|0|0|
|HK_local_transport|IDENTITY|17|5|3|3|0|0|0|
|HK_local_transport|IDENTITY|9|164|162|162|0|0|0|
|HK_local_transport|IDENTITY|18|33|16|16|0|0|0|
|HK_local_transport|IDENTITY|13|160|151|151|0|0|0|
|HK_local_transport|IDENTITY|21|4|0|0|0|0|0|
|HK_local_transport|IDENTITY|8|153|150|150|0|0|0|
|HK_local_transport|IDENTITY|3|157|135|135|0|0|0|
|HK_local_transport|ETF_FULL|4|165|142|141|0|1|0|
|HK_local_transport|ETF_FULL|5|1|0|0|0|0|0|
|HK_local_transport|ETF_FULL|6|1|0|0|0|0|0|
|HK_local_transport|ETF_FULL|22|21|15|14|0|1|0|
|HK_local_transport|ETF_FULL|20|66|16|17|1|0|0|
|HK_local_transport|ETF_FULL|1|113|107|107|0|0|0|
|HK_local_transport|ETF_FULL|2|182|154|153|0|1|0|
|HK_local_transport|ETF_FULL|15|123|122|122|0|0|0|
|HK_local_transport|ETF_FULL|14|60|58|57|0|1|0|
|HK_local_transport|ETF_FULL|12|41|33|33|0|0|0|
|HK_local_transport|ETF_FULL|10|150|113|111|0|2|0|
|HK_local_transport|ETF_FULL|11|61|18|18|0|0|0|
|HK_local_transport|ETF_FULL|19|2|0|0|0|0|0|
|HK_local_transport|ETF_FULL|16|8|3|3|0|0|0|
|HK_local_transport|ETF_FULL|7|21|21|21|0|0|0|
|HK_local_transport|ETF_FULL|0|7|5|5|0|0|0|
|HK_local_transport|ETF_FULL|17|5|3|3|0|0|0|
|HK_local_transport|ETF_FULL|9|164|162|162|0|0|0|
|HK_local_transport|ETF_FULL|18|33|16|16|0|0|0|
|HK_local_transport|ETF_FULL|13|160|151|151|0|0|1|
|HK_local_transport|ETF_FULL|21|4|0|0|0|0|0|
|HK_local_transport|ETF_FULL|8|153|150|149|0|1|0|
|HK_local_transport|ETF_FULL|3|157|135|135|0|0|0|
|HK_local_transport|ETF_DISC|4|165|142|141|0|1|0|
|HK_local_transport|ETF_DISC|5|1|0|0|0|0|0|
|HK_local_transport|ETF_DISC|6|1|0|0|0|0|0|
|HK_local_transport|ETF_DISC|22|21|15|14|0|1|0|
|HK_local_transport|ETF_DISC|20|66|16|17|1|0|0|
|HK_local_transport|ETF_DISC|1|113|107|107|0|0|0|
|HK_local_transport|ETF_DISC|2|182|154|153|0|1|0|
|HK_local_transport|ETF_DISC|15|123|122|122|0|0|0|
|HK_local_transport|ETF_DISC|14|60|58|57|0|1|0|
|HK_local_transport|ETF_DISC|12|41|33|33|0|0|0|
|HK_local_transport|ETF_DISC|10|150|113|111|0|2|0|
|HK_local_transport|ETF_DISC|11|61|18|18|0|0|0|
|HK_local_transport|ETF_DISC|19|2|0|0|0|0|0|
|HK_local_transport|ETF_DISC|16|8|3|3|0|0|0|
|HK_local_transport|ETF_DISC|7|21|21|21|0|0|0|
|HK_local_transport|ETF_DISC|0|7|5|5|0|0|0|
|HK_local_transport|ETF_DISC|17|5|3|3|0|0|0|
|HK_local_transport|ETF_DISC|9|164|162|162|0|0|0|
|HK_local_transport|ETF_DISC|18|33|16|16|0|0|0|
|HK_local_transport|ETF_DISC|13|160|151|151|0|0|1|
|HK_local_transport|ETF_DISC|21|4|0|0|0|0|0|
|HK_local_transport|ETF_DISC|8|153|150|149|0|1|0|
|HK_local_transport|ETF_DISC|3|157|135|135|0|0|0|
|HK_local_transport|MPC_DSM|4|165|142|141|0|1|0|
|HK_local_transport|MPC_DSM|5|1|0|0|0|0|0|
|HK_local_transport|MPC_DSM|6|1|0|0|0|0|0|
|HK_local_transport|MPC_DSM|22|21|15|14|0|1|0|
|HK_local_transport|MPC_DSM|20|66|16|16|2|2|1|
|HK_local_transport|MPC_DSM|1|113|107|107|0|0|0|
|HK_local_transport|MPC_DSM|2|182|154|152|0|2|0|
|HK_local_transport|MPC_DSM|15|123|122|122|0|0|0|
|HK_local_transport|MPC_DSM|14|60|58|53|0|5|0|
|HK_local_transport|MPC_DSM|12|41|33|33|0|0|0|
|HK_local_transport|MPC_DSM|10|150|113|111|0|2|0|
|HK_local_transport|MPC_DSM|11|61|18|18|0|0|1|
|HK_local_transport|MPC_DSM|19|2|0|0|0|0|0|
|HK_local_transport|MPC_DSM|16|8|3|3|0|0|0|
|HK_local_transport|MPC_DSM|7|21|21|21|0|0|0|
|HK_local_transport|MPC_DSM|0|7|5|5|0|0|0|
|HK_local_transport|MPC_DSM|17|5|3|3|0|0|0|
|HK_local_transport|MPC_DSM|9|164|162|162|0|0|0|
|HK_local_transport|MPC_DSM|18|33|16|15|0|1|0|
|HK_local_transport|MPC_DSM|13|160|151|151|0|0|0|
|HK_local_transport|MPC_DSM|21|4|0|0|0|0|0|
|HK_local_transport|MPC_DSM|8|153|150|149|0|1|0|
|HK_local_transport|MPC_DSM|3|157|135|133|0|2|0|

## 费用
GPU各8次实际驻留总272.310514654秒，CPU四次准备61.850206114秒；CPU fixture驻留7.703518271秒包含pipeline0.145757198秒，不相加。并行墙钟169.0565071105957秒。闭合历史326045.485630882秒，本轮后326317.796145536秒。复用旧结果计费0。全部独立记录及错误见AUDIT。
完整旧新混淆、原始/几何归一化margin正零负、31类分母/纠正破坏见RESULTS/CSV。极少n类连续值抑制，missing不填0；供体向错误类的重合不等于独立中心风险估计。


本轮四个最终checkpoint的CPU准备、四条真实零更新预检、四次统一评价均通过；最大正式求解残差6.694740186e−10。原MPC/DSM/组合全部最终指标与上一轮精确相同，OFF逐类n/recall及Acc与公开控制相同；IDENTITY所有分数逐元素相同、无预测变化。没有重训、修改模型或按val选参数。

**全头范数不是这次下降的主要解释。** 原公共能量占比 ISIC base/local分别0.275817%/0.266344%，HK为0.067157%/0.065654%。原ETF目标因此只较判别尺度大约0.138195%/0.133439%/0.033595%/0.032843%（精确比例见下表）。ETF_DISC与ETF_FULL在四个状态全部预测一致，包括错转错，六个分类指标全相同。不能把此前公共分量放大风险当作已经确认的故障。

**DSM方向与正则化的作用需要分开。** ISIC base零目标正则化相对原头BA+1.895009pp、纠正6/破坏0；原ETF相对原头+1.198546pp、纠正5/破坏1。ETF相对同γ零目标仅救1个新类，却破坏3个旧类，BA−0.696463pp。local_transport下两者BA/Acc/old/new相同，ETF只引起1个旧类错转错及F1+0.048151pp。所以ISIC原DSM的正结果不能称为ETF几何收益。

HK零目标相对原头BA为−0.364783/−0.410723pp（base/local），各破坏3个新类；加ETF目标相对零目标救回类3的2张、没有新增新类破坏，但仍比原控制少1张类8正确。ETF相对零目标整体纠正5/破坏3，各有旧类纠正3/破坏3；base BA−0.001196pp、local BA+0.060468pp，tail均下降−0.396825/−0.005411pp。这是存在局部方向贡献且总体不充分，不能概括成ETF完全没有作用，也不能把负结果全部归于正则化。

**MPC确实在混合不同类别方向，但并未验证供体是可靠中心。** HK类3主要供体为类4（权重约0.359、中心cos≈0.9767），原类3→4有20例，两控制加MPC各新增1例正确→4。类10的主要供体11/16也是实际混淆类，local中却分别救回1例；供体与混淆重合不必然有害。类14主要供体18/20/13，与新增破坏去向5不同；base/local却分别新增5/4例正确→5。类8的主要供体18/13/20也不同于新增破坏去向19。这说明全类联合解析重解和其他类头的变化参与了错误，不能只看目标类位移大小或最近供体就认定原因。类14有效供体数约12.768，无单供体坍缩。其位移沿18/20/13的cos约0.919/0.915/0.822；support heuristic只控制强度，仍无独立真实中心参照。

**原始margin下降不能全解读成几何分离下降。** HK local的MPC/REG_ZERO/ETF_FULL/组合新类原始margin平均差均为负（−0.001874/−0.015100/−0.003445/−0.005229），固定原最强竞争类的几何归一化margin平均差却全部为正（+0.000121/+0.001801/+0.001357/+0.001502）。平均几何改善仍伴随2/3/1/3例新类破坏，因此既存在分数尺度影响，也存在真实决策损失；不能用任一平均margin代替纠正/破坏计数。ISIC local归一化MPC/组合均负、REG_ZERO/ETF均正，四者新类各损失1/2/1/1例（顺序MPC/组合/REG_ZERO/ETF）。

本轮结论支持优先重新审视供体的类身份保持与全头耦合，以及正则强度与目标方向的分离；不支持仅修正公共范数便能恢复HK，也不支持按照当前val挑头。没有自动修改模块、调γ/η/温度或追加实验。

## 原头与尺度核对

完整机制分析、新类逐类变化及费用另见 ANALYSIS_ZH.md。

报告CPU pipeline总0.159609124秒，驻留未单测；本轮实际val推理曝光4242（四预检256+四评价3986），多个头共享特征，不重复算影像曝光。所有工程事件包含一次已结束CPU fixture命名例外，见ENGINEERING_AUDIT/ANALYSIS。
