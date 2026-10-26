Corpora used from Day 86 onward
===============================

text8_first_2M_words.txt.gz
    The first 2,000,000 words of text8 (Matt Mahoney, http://mattmahoney.net/dc/textdata.html):
    the first 100 MB of an English Wikipedia dump (March 2006), cleaned to lowercase letters a-z
    and single spaces. Downloaded from http://mattmahoney.net/dc/text8.zip on 2026-10-26; only
    this slice is kept, gzip-compressed (the full file is 100 MB, above GitHub's file limit).
    Wikipedia text is licensed CC BY-SA.
    Used by Days 86-89 (Word2Vec, embedding visualization, subword embeddings, GloVe).

tinyshakespeare.txt
    Andrej Karpathy's Tiny Shakespeare (about 40,000 lines from Shakespeare's plays, 1,115,394
    characters), from https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare,
    downloaded 2026-10-26, unchanged. Shakespeare's plays are in the public domain.
    Used by Day 90 (a tiny character-level GPT-style model) and Week 14.
