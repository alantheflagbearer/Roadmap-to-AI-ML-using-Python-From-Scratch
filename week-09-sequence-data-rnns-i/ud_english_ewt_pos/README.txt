Part-of-speech tagging data used by day62_many_to_many_tagger.py

SOURCE
  Universal Dependencies English Web Treebank (UD_English-EWT)
  https://github.com/UniversalDependencies/UD_English-EWT
  Silveira et al. (2014), "A Gold Standard Dependency Corpus for English", LREC 2014.
  Licence: Creative Commons Attribution-ShareAlike 4.0 International (see LICENSE.txt).

WHAT IS HERE (a reduced, modified copy -- shared under the same CC BY-SA 4.0 licence)
  Only two of the ten CoNLL-U columns are kept: the word form and its universal
  part-of-speech tag (UPOS). Multiword-token range lines (ids like "1-2"), empty
  nodes (ids like "8.1") and comment lines are dropped. Nothing else is changed.

  train.txt   all 2,001 sentences of en_ewt-ud-dev.conllu
  val.txt     the first 430 sentences of en_ewt-ud-test.conllu (cut at a document boundary)
  test.txt    the remaining 1,647 sentences of en_ewt-ud-test.conllu

  The treebank's own (much larger) training file is not used, to keep this a small
  "toy" dataset. The split follows the original files, so no document is shared
  between train and test.

FORMAT
  One token per line: word<TAB>tag. A blank line ends a sentence.
