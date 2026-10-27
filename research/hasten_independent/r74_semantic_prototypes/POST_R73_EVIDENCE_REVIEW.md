# Cross-run R67-R73 evidence review and R74 decision

All seven HK74002 six-stage runs complete and verified/delivered. Readonly saved aggregates, curves, class counts and diagnostics; no new model scoring or validation screening. Common SFT66.840041/64.876782/44.924938, H055.630100/56.454053/36.857416. Every frozen samplingprimary remains belowSFT. Same-seed original controls repeatedlyreproduce, engineeringgatespass, so do not claim a hidden runtimefault caused allnegative results.

| Run | Frozen sampling primary | mean/final/tail BA |
|---|---|---|
|R67|tars_grpo|66.115954/63.655574/44.737644|
|R68|raw_tars_grpo|65.295557/63.727152/43.586882|
|R69|fixed_tars_grpo|66.133591/63.554042/44.737644|
|R70|protected_tars_grpo|65.828378/62.006397/42.032016|
|R71|balanced_guard_grpo|65.494932/62.495759/42.962128|
|R72|compensated_grpo|64.899613/63.053406/43.866204|
|R73|hardnegative_grpo|65.776356/63.472624/44.482559|

Policyprioralignment failed; fixingvisualsupports failed forsampling, withfixedexactonlysmalltailpositive; natural/balancedCEdirectionguards worsenedoronlypartiallyrecovered; analyticbinarysignalcompensation producedactualsurrogateweightmax22402.95/KL.104034 andfailed; threelevelhardnegativereward added415allwronginformativegroupsbutfailedbothsamplingandexact. Exactexpectationhaslocalizedpositives(R72mean+.063901, R69tail+.324315vsSFT) withoutoverallgain. Therefore neither sparsegroupreward, samplingvariance, priorchoice, supportrefreshnorlocalriskprojectionisestablishedasuniqueexplanation. More rewardvariantscannotbejustifiedsolelybywantingGRPOtowin. Allcountsandnegative/secondaryevidenceretained, no bestarmorposthocgateadoption.

Structural untestedmechanism: few-shotvisualprototypeestimation uses asfewas3TRAINrows(classweightsrange3-736;someoppositefoldclasssupports1row). R69fixedsupportonlyfrozesanexistingvisualestimate; itdoesnotintroduceindependentpretrainingsemanticknowledge. CurrentclassifierprototypesarevisualTRAINmeans despiteavailablefrozenBiomedCLIPtextencoderandhumanclassnames. Assetinventoryconfirms existinglocal783705670Bcheckpoint/tokenizer/open_clip_config, no downloadneeded. Standardexistingprompttemplate is 'an endoscopic image of {}.' and existingorderedclassnames; do notsearchprompts/aliases oradddescriptions. Textsemantics maynotseparateoverlappingulcerative-colitisgradesorBarrettclasses, andpretrainingdomainalignmentmayfail. Thisisahypothesis,notsemanticqualityevidence.

R74 pseudocount-oneprototype: normalizen_support*normalizedvisualmean+normalizedfrozenclass_text. Oneunittextpseudo-observation is a predeclaredsimpleprecisionheuristic,notBayesianoptimalityorhyperparamsearch; positivecountsensurelargertextfractionforrareclasses. ApplyconsistentlytoTRAINoppositefoldsupports, fixedSFTrefsupportsandfinalfullhistoryprototypes; useactualsupportcountsforfolds,allTRAINcountsforinference. Onlyarrivedclassnamesencodedperstage, samefrozencheckpointandprompt. Zero-vectorcancellationfallbackvisualonly(numericalsafety). Frozenimageprefixunchangedandlast2qvLoRAonlylearned.

Fourstagewisearmsvisual_grpo,semantic_grpo(primary),visual_exact,semantic_exactwithoriginalbinaryTARSgoal. Addsemantic_sft_reference(nooptimizer)besideunchangedSFTreadout: distinguishreadoutpriorbenefitfromGRPOincrementalbenefit. AllnewheadssealbeforeanyVALincludingsemanticSFT. RequirecomparisonprimarytosemanticSFTaswellasoriginalSFT/visualcontrols,notjustH0. IfsemanticshelpswithoutRL, reportthatanddo notattributeit toGRPO. Thisisprototypecombinationwithknownshrinkageform,notclaimingnovelRLorforcingGRPOwin. Allsingle-seeddevelopmentlimitationsremain.
