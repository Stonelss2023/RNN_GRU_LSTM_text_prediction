def process_data(corpus_chars, num_chars=10000):
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[:num_chars]

    idx_to_char = list(set(corpus_chars))
    char_to_idx = {char: i for i, char in enumerate(idx_to_char)} #enumerate接受一个可迭代对象，每次迭代返回一个tuple（索引，值）
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]

    return corpus_chars, idx_to_char, char_to_idx, vocab_size, corpus_indices

"""
1. 去除空格符和换行符
2. 限制处理数据数量,只取前10000位
3. 创建不重复字符集合,并用list方式转化为列表
4. 字典推导式 → char键 + idx值
5. 取字典长度
6. 列表推导式 → 字典键对值 + 原始文本遍历
7. 返回 原始文本+不重复char列表+char:i键值对字典+字母表长度+原始文本索引"""


