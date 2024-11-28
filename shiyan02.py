import zipfile
import os
import re
from collections import defaultdict
import numpy as np
import streamlit as st
import pandas as pd

# 停用词列表
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', 'should', 'now'
])

# 解压 ZIP 文件
def unzip_dataset(zip_file_path, extract_to_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)

# 读取邮件内容
def read_emails_from_directory(directory):
    emails = []
    email_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        emails.append(f.read())
                        email_paths.append(file_path)
                except Exception as e:
                    st.warning(f"无法读取文件 {file_path}: {e}")
    return emails, email_paths

# 预处理文本
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [token for token in tokens if token not in STOP_WORDS]

# 生成词项词典
def generate_term_dictionary(emails):
    term_freq = defaultdict(int)
    for email in emails:
        tokens = preprocess_text(email)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            term_freq[token] += 1
    return {term: idx for idx, term in enumerate(sorted(term_freq.keys()))}

# 创建词项文档关联矩阵
def create_term_doc_matrix(emails, term_dictionary):
    num_terms = len(term_dictionary)
    num_docs = len(emails)
    term_doc_matrix = np.zeros((num_terms, num_docs), dtype=int)
    terms = list(term_dictionary.keys())  # 按顺序获取词项列表
    
    for doc_index, email in enumerate(emails):
        tokens = set(preprocess_text(email))
        for token in tokens:
            if token in term_dictionary:
                term_doc_matrix[term_dictionary[token], doc_index] = 1
    
    return term_doc_matrix, terms

# 创建倒排索引
def create_inverted_index(emails, term_dictionary):
    inverted_index = defaultdict(set)
    for doc_index, email in enumerate(emails):
        tokens = set(preprocess_text(email))
        for token in tokens:
            if token in term_dictionary:
                inverted_index[token].add(doc_index)
    return inverted_index

# 解析布尔查询（文档关联矩阵）
def parse_boolean_query_matrix(query, term_doc_matrix, terms):
    query = query.upper().split()
    result_docs = None
    current_op = 'AND'

    for term in query:
        if term in ['AND', 'OR', 'NOT']:
            current_op = term
        else:
            term_index = term_dictionary.get(term.lower())
            if term_index is not None:
                term_docs = set(np.where(term_doc_matrix[term_index] == 1)[0])
            else:
                term_docs = set()
            
            if current_op == 'AND':
                result_docs = result_docs & term_docs if result_docs is not None else term_docs
            elif current_op == 'OR':
                result_docs = result_docs | term_docs if result_docs is not None else term_docs
            elif current_op == 'NOT':
                result_docs = result_docs - term_docs if result_docs is not None else set(range(term_doc_matrix.shape[1])) - term_docs

    return sorted(result_docs) if result_docs else []

# 解析布尔查询（倒排索引）
def parse_boolean_query_inverted(query, inverted_index):
    query = query.upper().split()
    result_docs = None
    current_op = 'AND'

    for term in query:
        if term in ['AND', 'OR', 'NOT']:
            current_op = term
        else:
            term_docs = inverted_index.get(term.lower(), set())

            if current_op == 'AND':
                result_docs = result_docs & term_docs if result_docs is not None else term_docs
            elif current_op == 'OR':
                result_docs = result_docs | term_docs if result_docs is not None else term_docs
            elif current_op == 'NOT':
                result_docs = result_docs - term_docs if result_docs is not None else set()

    return sorted(result_docs) if result_docs else []

# Streamlit 界面
st.set_page_config(page_title="布尔检索系统", layout="wide")


# 侧边栏导航
# 定义导航栏逻辑
with st.sidebar:
    st.title("布尔检索系统")
    st.markdown("---")  # 分割线

    # 导航栏中的动态页面切换
    current_page = st.radio("导航", ["首页", "解压数据集", "倒排索引文档", "布尔检索","关于"], label_visibility="visible")

    st.markdown("---")
    # 添加外部链接和信息
    st.markdown("📄 [Streamlit 官方文档](https://docs.streamlit.io)")
    st.markdown("📧 联系开发者: [3362367257@qq.com](mailto:3362367257@qq.com)")


# 首页
if current_page == "首页":
    # 标题居中样式
    st.markdown(
        '<h2 style="text-align:center;">♏ 布尔检索系统简介</h2>',
        unsafe_allow_html=True,
    )

    # 简介内容部分
    st.write(
        """
        欢迎使用布尔检索系统！本系统结合布尔逻辑与现代文本检索技术，为用户提供精准、高效、灵活的**邮件文档检索**解决方案。

        ##### 💻 系统简介：
        - **核心原理**：基于布尔逻辑的检索方法，通过 AND、OR、NOT 等操作符实现多条件查询。
        - **技术支持**：采用倒排索引技术，提高大规模文档集合的检索效率。
        - **多场景适用**：支持跨目录、跨主机的文档管理与检索，满足多样化需求。

        ##### 💡 系统亮点：
        - **精准搜索**：结合多个布尔逻辑的词项查询条件，准确给出查询结果并支持在线预览；
        - **快速响应**：通过倒排索引技术，确保大数据量环境下的实时响应；
        - **直观体验**：结合文字和浅显易懂的各种符号，操作简单，提供用户友好的交互界面与清晰的检索结果展示；
        - **了解知识**：本系统提供了在线了解布尔检索相关知识，让用户在进行邮件文档检索的过程中仍能实时学习；
        - **随进随用**：本系统不需要注册登录，不收取任何费用，随时登录随时使用。

        ##### 🚀 系统价值：
        - **提升效率**：缩短信息筛选时间，让用户专注于核心任务；
        - **增强精确度**：精准匹配需求文档，避免遗漏重要信息；
        - **支持深度查询**：无论是简单搜索还是复杂逻辑组合，都可以高效实现。

        通过本系统，您将体验高效的文档搜索与管理，助力快速决策和高效工作。立即开始探索吧！
        """
    )

    # 添加分割线
    st.divider()

    # 系统功能标题
    st.markdown(
        '<h4 style="text-align:center; color:#000000;">🧰 系统功能</h4>',
        unsafe_allow_html=True,
    )

    # 卡片布局
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown(
            """
            <div style="background-color:#f0f0f5; padding:20px; border-radius:10px; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);">
                <h5 style="text-align:center;">📂 数据解压</h5>
                <p style="text-align:left;">&#8226; 根据用户在主机中的存储路径（输入时不加引号）解压文档数据集。</p>
                <p style="text-align:left;">&#8226; 在后续进行文档检索的过程中，务必先对数据集进行解压。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background-color:#f0f0f5; padding:20px; border-radius:10px; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);">
                <h5 style="text-align:center;">📖 倒排索引</h5>
                <p style="text-align:left;">&#8226; 提供倒排索引文档的在线浏览和下载。</p>
                <p style="text-align:left;">&#8226; 支持在倒排索引文档中对相应词项进行检索。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div style="background-color:#f0f0f5; padding:20px; border-radius:10px; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);">
                <h5 style="text-align:center;">🔍 布尔检索</h5>
                <p style="text-align:left;">&#8226; 使用布尔操作符精确检索所需要的文档。</p>
                <p style="text-align:left;">&#8226; 支持 AND、OR、NOT 等操作。</p>
                <p style="text-align:left;">&#8226; 提供检索结果的统计信息及文档内容在线预览。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 再次分割线
    st.divider()


    # 页面标题
    st.markdown('<h4 style="text-align:center;">📚 布尔检索相关知识了解</h4>', unsafe_allow_html=True)

    # 创建选项卡
    tabs = st.tabs(["布尔逻辑", "词项文档关联矩阵", "倒排索引文档", "基于关联矩阵的布尔检索", "基于倒排索引的布尔检索", "两种布尔检索的优缺点", "实际应用"])

    # 布尔逻辑
    with tabs[0]:
        st.markdown('<h4 style=";">布尔逻辑</h4>', unsafe_allow_html=True)
        st.write("""
            布尔逻辑是信息检索的理论基础，主要通过以下操作符构建复杂的查询条件：
            - **AND**：仅返回同时包含所有查询条件的文档。例如，`数据 AND 检索` 返回包含“数据”和“检索”的文档。
            - **OR**：返回包含任意查询条件的文档。例如，`数据 OR 检索` 返回包含“数据”或“检索”的文档。
            - **NOT**：排除包含指定查询条件的文档。例如，`数据 NOT 检索` 返回包含“数据”但不包含“检索”的文档。
            
            **特点**：
            - 简单高效：操作符易理解，适合小规模数据检索。
            - 灵活组合：支持嵌套条件，如 `(数据 AND 检索) OR (文档 AND 索引)`。
            
            **应用场景**：
            - **学术检索**：快速筛选相关文献，例如 `"深度学习" AND "自然语言处理" OR "图像识别"`。
            - **搜索引擎**：实现复杂的关键词组合查询。
        """)

    # 文档词项关联矩阵
    with tabs[1]:
        st.markdown('<h4 style="; ">词项文档关联矩阵</h4>', unsafe_allow_html=True)
        st.write("""
            **定义**：
            文档词项关联矩阵是一种结构化表示，用于记录词项与文档的关系：
            - 行表示文档；
            - 列表示词项；
            - 单元格的值可以是布尔值（是否存在）或频率（出现的次数）。

            **构建步骤**：
            1. **文档预处理**：对文档进行分词、去停用词、提取词干等操作。
            2. **词项列表生成**：提取所有独立词项，创建词项列表。
            3. **构建矩阵**：逐一记录每个词项与文档的关联关系。
        """)


        st.write("**构建词项文档关联矩阵伪代码：**")
        st.code("""
        # 输入：一组文档集合
        documents = ["文档1内容", "文档2内容", ..., "文档N内容"]
        
        # 1. 文档预处理
        preprocessed_documents = preprocess(documents)  # 调用分词、去停用词等
        
        # 2. 构建词项列表
        term_list = extract_unique_terms(preprocessed_documents)
        
        # 3. 构建关联矩阵
        initialize_matrix(len(documents), len(term_list))  # 初始化矩阵
        for each_document in preprocessed_documents:
            for each_term in term_list:
                if term_exists_in_document(each_term, each_document):
                    set_matrix_value(document_index, term_index, 1)  # 布尔值
        
        # 输出结果矩阵
        print_matrix()
        """, language="text")


        st.write("""
            **示例**：
            假设有以下文档：
            - Document 1: "Boolean logic is the foundation of retrieval systems."
            - Document 2: "An inverted index is a key component of Boolean retrieval."
            - Document 3: "Boolean operators support AND, OR, and NOT."

            以下展示如何构建词项文档关联矩阵：
        """)



        st.write("**1. 文档预处理示例：**")
        st.code("""
        输入文档：
        Document 1: "Boolean logic is the foundation of retrieval systems."
        Document 2: "An inverted index is a key component of Boolean retrieval."
        Document 3: "Boolean operators support AND, OR, and NOT."

        步骤：
        - 分词：将每个文档拆分成单独的词项。
        - 去停用词：移除诸如“is”、“the”、“of”等高频无意义词。
        - 提取词干：将词项简化为基础词形，如“retrieval”和“retrieve”。

        停用词列表示例：
        ["is", "the", "of", "and", "or", "an"]

        预处理后：
        Document 1: ["boolean", "logic", "foundation", "retrieval", "systems"]
        Document 2: ["inverted", "index", "key", "component", "boolean", "retrieval"]
        Document 3: ["boolean", "operators", "support", "not"]
        """, language="text")

        st.write("**2. 词项列表生成示例：**")
        st.code("""
        合并所有文档的词项后，提取唯一的词项列表：
        ["boolean", "logic", "foundation", "retrieval", "systems", "inverted", "index",
         "key", "component", "operators", "support", "not"]

        词项列表（按字母顺序排列）：
        ["boolean", "component", "foundation", "index", "inverted", "key", "logic",
         "not", "operators", "retrieval", "support", "systems"]
        """, language="text")

        
        st.write("""
            **3. 最终构建的词项文档关联矩阵如下所示（示例结果）：**
            |      | boolean | component | foundation | index | inverted | key | logic | not | operators | retrieval | support | systems |
            |------|---------|-----------|------------|-------|----------|-----|-------|-----|-----------|-----------|---------|---------|
            | 文档1 | 1       | 0         | 1          | 0     | 0        | 0   | 1     | 0   | 0         | 1         | 0       | 1       |
            | 文档2 | 1       | 1         | 0          | 1     | 1        | 1   | 0     | 0   | 0         | 1         | 0       | 0       |
            | 文档3 | 1       | 0         | 0          | 0     | 0        | 0   | 0     | 1   | 1         | 0         | 1       | 0       |
        """)


    # 倒排索引文档
    with tabs[2]:
        st.markdown('<h4 style="; ">倒排索引文档</h4>', unsafe_allow_html=True)
        st.write("""
            **定义**：
            倒排索引是一种高效的文本检索数据结构，用于记录每个词项在哪些文档中出现及其位置信息。

            **构建步骤**：
            1. 对文档集合进行分词和预处理（包括去停用词和提取词干）。
            2. 为每个词项创建倒排列表，记录文档ID及出现次数。
            3. 将倒排列表存储为字典或树形结构。

            **优点**：
            - 高效支持大规模文档检索；
            - 占用内存小，适合静态数据。
        """)

        st.write("**倒排索引构建伪代码：**")
        st.code("""
        # 输入：一组文档集合
        documents = ["文档1内容", "文档2内容", ..., "文档N内容"]

        # 1. 文档预处理
        preprocessed_documents = preprocess(documents)  # 包括分词、去停用词、提取词干等

        # 2. 初始化倒排索引
        inverted_index = {}

        # 3. 遍历文档构建倒排列表
        for doc_id, document in enumerate(preprocessed_documents):
            for term in document:
                if term not in inverted_index:
                    inverted_index[term] = set()  # 初始化倒排列表
                inverted_index[term].add(doc_id)

        # 输出倒排索引
        print(inverted_index)
        """, language="text")

        st.write("""
            **示例**：
            假设有以下文档：
            - Document 1: "Boolean logic is the foundation of retrieval systems."
            - Document 2: "An inverted index is a key component of Boolean retrieval."
            - Document 3: "Boolean operators support AND, OR, and NOT."

            **预处理后：**
            - Document 1: ["boolean", "logic", "foundation", "retrieval", "systems"]
            - Document 2: ["inverted", "index", "key", "component", "boolean", "retrieval"]
            - Document 3: ["boolean", "operators", "support", "not"]

            **构建倒排索引：**
            每个词项的倒排列表记录如下：
            - "boolean": {Doc 1, Doc 2, Doc 3}
            - "logic": {Doc 1}
            - "retrieval": {Doc 1, Doc 2}
            - "inverted": {Doc 2}
            - "operators": {Doc 3}

        """)


    # 基于关联矩阵的布尔检索
    with tabs[3]:
        st.markdown('<h4 style="; ">基于关联矩阵的布尔检索</h4>', unsafe_allow_html=True)
        st.write("""
            **实现步骤**：
            1. 根据查询条件识别词项；
            2. 从关联矩阵中提取对应词项的列；
            3. 通过逻辑运算对列向量组合，获取符合条件的文档ID。

            **示例：**
            - 查询条件："boolean AND retrieval"
            - 对应矩阵中提取的列：
              - "boolean": [1, 1, 1]
              - "retrieval": [1, 1, 0]
            - 逻辑运算："AND"：
              - 结果：[1, 1, 0]
            - 匹配文档ID：Doc 1, Doc 2

            **优缺点**：
            - 优点：结构清晰，适合小规模数据。
            - 缺点：效率较低，内存占用大。
        """)

    # 基于倒排索引的布尔检索
    with tabs[4]:
        st.markdown('<h4 style=";">基于倒排索引的布尔检索</h4>', unsafe_allow_html=True)
        st.write("""
            **实现步骤**：
            1. 根据查询条件识别词项，定位倒排列表；
            2. 通过布尔操作符对倒排列表进行逻辑运算；
            3. 返回匹配文档ID并提供文档内容预览。

            **示例：**
            - 查询条件："boolean AND retrieval"
            - 倒排列表：
              - "boolean": {Doc 1, Doc 2, Doc 3}
              - "retrieval": {Doc 1, Doc 2}
            - 逻辑运算："AND"：
              - 结果：{Doc 1, Doc 2}
            - 匹配文档ID：Doc 1, Doc 2

            **优缺点**：
            - 优点：适合大规模数据检索，速度快，占用内存小；
            - 缺点：索引构建耗时，动态更新成本高。
        """)
    # 两种布尔检索的优缺点
    with tabs[5]:
        st.markdown('<h4 style="; ">两种布尔检索的优缺点</h4>', unsafe_allow_html=True)
        st.write("""
            **布尔检索是信息检索中的基础方法，不同实现方式有其适用场景和局限性。以下是基于关联矩阵和倒排索引两种方法的对比分析：**

            | 检索方式                 | 优点                                                                 | 缺点                                                                 |
            |--------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
            | **基于关联矩阵**         | 1. 结构直观，易于理解和实现；                                       | 1. 占用内存较大：尤其在词项和文档数量较多时，矩阵会非常稀疏；        |
            |                          | 2. 适合小规模数据分析：处理数据量较少时无需额外构建复杂索引结构；    | 2. 检索效率低：逻辑运算需要对整列数据进行处理，速度慢；              |
            |                          | 3. 灵活性高：无需提前构建索引，可直接基于文档内容进行操作；          | 3. 不适合动态更新：每次查询均需重新计算关联矩阵，性能下降。           |
            | **基于倒排索引**         | 1. 检索速度快：通过直接访问倒排列表，查询性能显著提升；              | 1. 索引构建耗时长：需要扫描所有文档并建立完整索引；                  |
            |                          | 2. 适合大规模数据：索引占用内存小，即使文档量大也能快速响应；         | 2. 动态更新成本高：当文档集合有变化时，需要重新更新索引；            |
            |                          | 3. 支持复杂查询：结合布尔运算符（AND、OR、NOT），能处理多词项查询；    | 3. 不适合实时数据：更新或删除操作较频繁时，索引构建的开销较大。       |

            **总结**：
            - **关联矩阵**更适合用于小规模数据的原型验证和初步研究。
            - **倒排索引**是目前主流搜索引擎和检索系统的核心方法，广泛应用于实际生产环境。
        """)

    # 实际应用
    with tabs[6]:
        st.markdown('<h4 style="; ">实际应用</h4>', unsafe_allow_html=True)
        st.write("""
            **布尔检索的实际应用领域广泛，特别适合需要快速过滤和定位内容的场景：**

            1. **学术研究**：
               - 学术数据库和文献管理工具中，通过布尔检索快速定位特定主题的论文。
               - 结合 AND/OR/NOT 运算符实现多条件查询，例如查找某领域的某特定年份的研究文献。

            2. **搜索引擎**：
               - 搜索引擎的早期实现，如 WAIS 和 Gopher 等，采用了布尔检索作为核心算法。
               - 用户输入类似 "Python AND Data Science" 的查询时，通过布尔逻辑返回相关结果。

            3. **电子邮件管理**：
               - 在邮件系统中，通过布尔查询按主题、发件人、日期等条件筛选邮件。
               - 例如，筛选所有包含"合同"但不包含"草稿"的邮件。

            4. **企业内容管理**：
               - 对企业内部文件或记录进行分类和检索，快速定位特定的合同、报告或客户资料。

            5. **法律领域**：
               - 检索法律文件、案例记录或法庭档案时，布尔检索被广泛应用以定位相关法律条文和案例支持。

            6. **医疗信息系统**：
               - 医生和研究人员通过布尔检索快速查找包含某些疾病、药物或治疗方案的病例数据或研究报告。

            7. **电子商务**：
               - 通过布尔逻辑筛选产品，比如查找所有包含"智能手机"且价格在一定范围内的商品。

            **布尔检索的灵活性**：
            通过布尔检索，用户可以根据需求自由组合查询条件，大幅提高查询的精确度和相关性，在各种场景中表现出极高的适用性。
        """)

     # 再次分割线
    st.divider()

    # 页面底部：联系信息和版权声明
    st.markdown("""
       <footer style="text-align:center;">
           <p>📬 如有任何问题，请联系：3362367257@qq.com</p>
           <p>© 2024 信息检索技术——嵇存老师 | dx</p>
       </footer>
        """, unsafe_allow_html=True)




# 第一步：解压数据集
elif current_page == "解压数据集":
    st.markdown('<h3 style="text-align:center;">♏解压数据集</h3>', unsafe_allow_html=True)

    zip_file_path = st.text_input("✍☞ZIP 文件路径:", "")
    extract_to_dir = st.text_input("✍☞解压路径:", "")

    if st.button("解压数据集"):
        if zip_file_path and extract_to_dir:
            try:
                unzip_dataset(zip_file_path, extract_to_dir)
                st.success("解压成功！")
                
                # 将解压路径存入 session_state
                st.session_state['extract_to_dir'] = extract_to_dir
                
                # 读取邮件内容
                emails, email_paths = read_emails_from_directory(extract_to_dir)
                st.session_state['emails'] = emails  # 保存邮件数据
                st.session_state['email_paths'] = email_paths  # 保存邮件路径

                # 显示读取邮件数量
                if emails:
                    st.write(f"共读取了 {len(emails)} 封邮件。")
                else:
                    st.warning("没有读取到邮件数据。")

            except Exception as e:
                st.error(f"解压失败: {e}")
        else:
            st.warning("请提供有效的 ZIP 文件路径和解压路径。")
    else:
        st.warning("请提供有效的 ZIP 文件路径和解压路径。")


# 第二步：倒排索引文档
elif current_page == "倒排索引文档":
    # 居中显示标题
    st.markdown('<h3 style="text-align:center;">♏倒排索引文档</h3>', unsafe_allow_html=True)

    # 确保第一步完成并且解压路径有效
    if 'extract_to_dir' in st.session_state and st.session_state.extract_to_dir:
        extract_to_dir = st.session_state.extract_to_dir
        if os.path.exists(extract_to_dir):
            emails, email_paths = read_emails_from_directory(extract_to_dir)

            if emails:
                term_dictionary = generate_term_dictionary(emails)
                term_doc_matrix, terms = create_term_doc_matrix(emails, term_dictionary)
                inverted_index = create_inverted_index(emails, term_dictionary)

                # 创建倒排索引表格
                inverted_index_df = pd.DataFrame(
                    [(term, ",".join(map(str, docs))) for term, docs in inverted_index.items()],
                    columns=["Term", "Document IDs"]
                )

                # 设置样式使表格更宽
                st.markdown("""
                    <style>
                    .wide-table-container {
                        max-width: 90%;  /* 表格宽度占页面 90% */
                        margin: 0 auto; /* 居中表格 */
                    }
                    </style>
                """, unsafe_allow_html=True)

                
                # 在样式容器中显示表格
                st.markdown('<div class="wide-table-container">', unsafe_allow_html=True)
                st.dataframe(inverted_index_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # 提示文字
                st.markdown("""
                    **✿提示**: 
 """, unsafe_allow_html=True)
                
                                
                st.write(f"- 您可以将鼠标放置表格中☝，点击右上角的“↓”下载该倒排索引文档。")
                st.write(f"- 您可以将鼠标放置表格中☝，点击右上角的“⌕”在倒排索引文档中查找相应词项。")
                st.write(f"- 您可以拖动“←║→”改变表格列宽")



            else:
                st.warning("没有找到任何邮件。")
        else:
            st.warning("请先解压数据集并加载邮件。")
    else:
        st.warning("请先解压数据集并加载邮件。")




elif current_page == "布尔检索":
    st.markdown('<h3 style="text-align:center;">♏布尔检索</h3>', unsafe_allow_html=True)

    if 'emails' in st.session_state and st.session_state.emails:
        emails = st.session_state.emails
        email_paths = st.session_state.email_paths
        term_dictionary = generate_term_dictionary(emails)
        term_doc_matrix, terms = create_term_doc_matrix(emails, term_dictionary)
        inverted_index = create_inverted_index(emails, term_dictionary)

        # 总邮件数展示
        total_emails = len(emails)
        st.info(f"当前共有 {total_emails} 封邮件可以检索。")

        # 检索方式选择
        st.markdown('<p style="font-size:16px; font-weight:bold;">🎯 选择布尔检索方式:</p>',unsafe_allow_html=True)
        search_method = st.radio("",["文档关联矩阵", "倒排索引"], label_visibility='collapsed')

        # 使用 st.markdown 来增加样式并缩小上下间距
        st.markdown(
            '''
            <style>
            .query-text {
                font-size: 16px;
                font-weight: bold;
                margin-top: 10px;
                margin-bottom: 0px;  /* 调整上下间距，减少底部间距 */
            }
            </style>
            <p class="query-text">🔍 请输入布尔查询内容 (支持 AND, OR, NOT):</p>''',unsafe_allow_html=True)

        # 创建文本输入框
        query_input = st.text_input("")
        
        if st.button("搜索"):
            # 执行检索
            if search_method == "文档关联矩阵":
                st.session_state.results = parse_boolean_query_matrix(query_input, term_doc_matrix, terms)
            else:
                st.session_state.results = parse_boolean_query_inverted(query_input, inverted_index)

            if st.session_state.results:
                st.success(f"共找到 {len(st.session_state.results)} 封匹配的邮件。")

                # 创建选项卡
                tabs = st.tabs(["结果概览", "详细预览"])

                # 选项卡 1: 检索结果概览
                with tabs[0]:
                    st.markdown('<h2 style="font-size:16px; font-weight:bold;">🚀 检索结果概览</h2>', unsafe_allow_html=True)

                    result_data = {
                        "文档ID": st.session_state.results,
                        "文档路径": [email_paths[doc_id] for doc_id in st.session_state.results]
                    }
                    result_df = pd.DataFrame(result_data)
                    st.dataframe(result_df)

                     # 添加统计数据摘要（可选）
                    st.markdown('<h2 style="font-size:16px; font-weight:bold;">🔢 统计信息</h2>', unsafe_allow_html=True)
                    st.write(f"- 匹配邮件占总邮件的比例: <u>*{len(st.session_state.results) / total_emails:.2%}*</u>", unsafe_allow_html=True)
                    st.write(f"- 检索方法: <u>*{search_method}*</u>", unsafe_allow_html=True)



                # 选项卡 2: 详细预览
                with tabs[1]:
                    st.markdown('<h2 style="font-size:16px; font-weight:bold;">🚀 详细预览</h2>', unsafe_allow_html=True)
                    for doc_id in st.session_state.results:
                        with st.expander(f"文档ID {doc_id} - 点击展开预览", expanded=False):
                            st.markdown(f"**📂 文档路径**: {email_paths[doc_id]}")
                            st.markdown('<h2 style="font-size:16px; font-weight:bold;">📖 文档内容</h2>', unsafe_allow_html=True)
                            st.text(emails[doc_id])  # 显示邮件内容

            else:
                st.warning("没有找到匹配的邮件，请调整查询条件重试。")

    else:
        st.warning("请先解压数据集并加载邮件。")


# 关于，跳转到源代码
elif current_page == "关于":
    st.markdown('<h3 style="text-align:center;">♏关于</h3>', unsafe_allow_html=True)

    st.markdown(
        """
        **技术说明：**
        - 使用 **Streamlit** 框架开发，快速构建交互式 Web 应用，支持实时交互和动态数据展示。
        - 集成 **Pandas** 用于数据处理和表格展示，方便数据统计和分析。
        - 采用布尔解析算法，支持复杂的查询表达式解析（如 `AND`, `OR`, `NOT`）。
        - 模块化代码设计，易于扩展和维护。
        - 使用 Python 语言及相关生态工具构建，兼具高效性与灵活性。

        """
    )
    st.markdown("🔗 [查看源代码](https://github.com/example/repo)")
