# R117: tailness-proportional query allocation with complete importance correction

Prospective independent diagnostic afterR116: strict fixed-reference protection kept head but accepted only29/256projected and13/256shrunk parameter updates, losing useful tail gain. Do not tune itsdelta/angle/lr. Pre-VALR114/R116TRAIN analyses identified extreme per-queryTARSweights(max63.84),rarequeryunderexposureandzero-responseopportunity. This experiment changes whereTARSlearning effort is allocated, without changing the fixed-policy expected originalTARSsurrogate or referencepenalty. It is a mechanism diagnostic, not a claimed new final/top-tier method or completedcontinualresult.

## Objective and allocation

Let p_c=n_c/N be naturalTRAIN classmass and w_c the fixedR114SFT-derived originalTARSweight. Set Z=sum p_c*w_c=1.4181295401301217, q_c=p_c*w_c/Z. Sample class accordingtoq,then uniformTRAINidentitywithinclass,withreplacement. Allqpositive. Let rho_c=p_c/q_c=Z/w_c. At a fixed current/behavior/reference policy, E_q[rho*(-w*GRPO_surrogate+beta*KL)] equals E_p[-w*GRPO_surrogate+beta*KL], including the original ratio clipping and within-querygroup normalization. Both terms require correction. Normalize each loss by fixedquery/responsecount, notsumimportanceweights. Globalgradientclipping andAdamupdatesare nonlinear,so their expectedupdates/trainingpaths neednotmatch.

Surrogatecoefficientrho*w=Zisconstant, reducing maxperquery coefficient63.84to1.4181 while retaining the expectedscale1.4181. Among allfullsupportsamplers, max_c(p_c*w_c/q_c)>=sum p_c*w_c=Z; thisq attainsthebound. This is standardimportanceallocation math, not a novelty claim. It increases opportunity for rare reward discovery, but cannot guarantee success where canonicalprobabilityisnegligible. OriginalTARSstillactsaftergroupnormalization; do notscale rawrewards.

## Frozen model/data/arms

StaticHK23,TRAIN6821/VAL1698/TEST0,seed74005commonR114SFT/LoRAr8alpha16/Qwen2.5-VL-3B/224features unchanged, all23classesvisible. Fixed originalTARSweights, frozenSFTreference, noextraheadanchorconstraint, no newbootstrap/visualencoder/resolutionchange. Existingcaches/SFTread-only. The originalpaperlargerresolution/fullnonvisionupdate/longerschedule differencesremain disclosed; notanexactreproduction.

Four newarms, each256outerupdates/8queries,G8forRL,independentstate/optimizerfromidenticalSFT. Allfourshareoneq-sampled256x8querytape(seed11774005),andRLgenerationseed11480000+k. Response sampling is onpolicy foreacharm, same seed doesnotimplyidenticalresponses. Noresampling/filteringofconstantrewardgroups orforcedcorrectanswers.

ALLOC_TARS_FULL_IS(primary):PGcoefficientrho*w=Z,KLcoefficientrho. ALLOC_TARS_PG_ONLY_IS: samePGcoefficientZ,butKLcoefficient1; ablates the otherwiseoftenomitted referencepenaltycorrection and thus changes thetargetobjective. ALLOC_GRPO_FULL_IS: PGcoefficientrho,KLcoefficientrho; removesTARSreweightingwhileusingthesameallocation/feedback. ALLOC_CE_FULL_IS:target-nameCEcoefficientrho*w=Z,noKL,query-budgetmatchedbutfewerresponseforwardsanddifferentregularization. Do notclaimthisisolatesMonteCarlonecessityorcomputeequivalence. This is afalsifying supervisedcontrol, not baselineoptimization.

ReuseR116sealedPLAIN_TARS256 endpointas naturalqueryTARSanchor: sameSFT/reference/hyperparameters/responsebudgetbutoriginalquerydistribution. Do notretrainit, countasanewreplicateorclaimitsquerytapeidenticaltothenewsampler. No post-hocbestepoch/seedselection.

AllRLhyperparametersunchanged:G8,strictcanonicalnamebinaryreward,temperature1/max24,groupstd+1e-6,oneepoch,ratio clip.2,baseKL.04,AdamW2e-5/wd0,gradclip1,micro2. Frozenbehaviorlogpsrecomputedinsameoptimizationpath,initialratioexact1; BF16cachedsamplingdifferencesretained. WeightedCEuses8labels/update,micro2,samelr/updates,noKL. Query importance/TARSweights detached. Full-allocationlossuses mean over64responses,notself-normalizedIS. Publishbothunweightedandimportance-weightedKL,realizedclassquerycounts,constantgroups,gradientclippingandresponsecosts.

## Readout/gates

SealallfourbeforeanynewVALinference; samegreedyreadoutandprivateper-examplearrays,23correct/count,micro/macro/head(n>=250)/tail(n<250)/tiny(n<25). MainversuscommonSFT:macro>=+1pp,tail>=+2pp,head>=-1pp; versusALLOC_GRPO_FULL_IS:macro>0ANDtail>0; versusreusednaturalPLAIN_TARS256:head>=0ANDtail>0; versusALLOC_CE_FULL_IS:macro>=0ANDtail>0. ReportallcomponentsandPG-onlyablationevenifconjunctionfails. Tinybenefitnotpresumed. Expectedobjectiveequivalencedoesnotguaranteeaccuracyimprovement. Onehistoricaldevelopmentseed/VALisnotindependentclinicalconfirmationorfullcontinuallearning.

## Admission/resources/delivery

Originalclockbeginsatpreparation,allretryshareit,harddeadlineNov9 12:45+08and<=24h. NativeusesexactlytwofixedTRAINimages0/1,real3RL+1CEbackward/optimizerpaths,strictcommonSFTrestore,same-policyprobabilitycheck,independentclasswiseexpectedobjective/gradientISfixtureincludingKLcorrection. Discardnativetraining. Fullmatrixforecastuses1.5xprofilemarginplus30minreserve,andmustfitremainingclock;noperformancepilot,nosilentmatrixshortening.

Onlypro5000GPU1,neutralargv,independent2sguard. Cumulativenewphysical<10,000,000,000B acrossR114/R115/R116/R117andlocalcopies,noreset. Retainedrootsabout2.34GB,newexpected<100MB,readonlyfeature/weightreuse,no download. Guardaddsallretainedbaseline+local200MB,newroot,prestop9GB;freeFS>=1GB/start>=5GB,mountread/writeprobe,GPUreserved<56GiB,groupRSS<48GB,CPU4,file4GB. Preserveallnegative/failure/costevidenceandoriginalclocks,nohistoric/otherprocesschanges. Finalsavedverification/source-onlysafeaggregateGitHubdeliveryusingexplicit7897proxy/remoteSHA/anonymous200;samematching30minresearchtaskupdatedratherthanduplicated. Ifsignalbecomesrobust,prospectivelyconfirmcontinualarrival/boundedmemorywithfreshgenericinit;donotcallthisstaticdiagnosticafinalnovelmethod.
