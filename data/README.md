The data for the EMNLP 2023 paper ```Exploring Jiu-Jitsu Argumentation for Writing Peer Review Rebuttals```

# Dataset
The dataset is contained in this folder
----------------------------------------
The top level are the attitude roots:  
* Arg_other  
* Asp_clarity
* Asp_meaningful-comparison
* Asp_motivation-impact
* Asp_originality
* Asp_replicability
* Asp_substance

The attitude roots are categorized into the following sub-olders:  
* actions (rebuttal actions such as rebuttal_answer, etc.) relevant to that attitude root.
	* Each file has the format review'\t'rebuttal.   
* rebuttal (the rebuttal sentences for the particular attitude theme)
	* Each file has the format review'\t'rebuttal.  
* review (all the review sentences relevant to this attitude root).  

The top level also has the folder ```canonical_rebuttals_and_descs```. This contains the following files. ('Aspects' signify attitude roots and  'Sections' signify themes)
* all_canonical_rebuttals_scores.tsv (contains all the canonical rebuttals with scores [tab] seperated)
	* Has the format [aspects '\t' sections '\t' action '\t' canonical_rebuttal '\t' scores]
* all_canonical_rebuttals.tsv (contains only the canonical rebuttals)
	* Has the format [aspects '\t' sections '\t' action '\t' canonical_rebuttal]
* all_cluster_descs.tsv (contains all the cluster descs)
	* Has the format [aspects '\t' sections '\t' descs]

