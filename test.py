import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="导航栏示例",
    layout="wide",  # 控制页面布局为宽屏
)

# 定义导航栏逻辑
with st.sidebar:
    # 添加图片和标题
    st.image("https://via.placeholder.com/150", width=150, caption="示例 Logo")
    st.title("导航栏示例")
    st.markdown("---")  # 分割线

    # 导航栏中的动态页面切换
    current_page = st.radio("选择页面：", ["首页", "功能页", "关于"], label_visibility="visible")
    
    # 多级菜单（折叠菜单）
    with st.expander("高级设置"):
        st.slider("调节参数A：", 0, 100, 50)
        st.slider("调节参数B：", 1, 10, 5)
        st.markdown("高级功能说明...")

    st.markdown("---")
    # 添加外部链接和信息
    st.markdown("📄 [访问文档](https://docs.streamlit.io)")
    st.markdown("📧 [联系我们](mailto:example@example.com)")

# 动态展示内容
if current_page == "首页":
    st.header("欢迎来到首页！")
    st.markdown(
        """
        **布尔检索简介：**
        布尔检索是一种基于布尔逻辑的文本检索技术，通过与、或、非等操作符来匹配文档与查询条件。
        """
    )
    st.video("https://www.youtube.com/watch?v=J0Aq44Pze-w")  # 插入讲解倒排索引的视频
elif current_page == "功能页":
    st.header("功能页")
    st.write("这里展示了各种功能的实现：")
    st.slider("功能参数调整", 0, 100, 25)
    st.checkbox("功能选项 1")
    st.checkbox("功能选项 2")
elif current_page == "关于":
    st.header("关于")
    st.markdown(
        """
        **应用信息：**
        - 本项目演示了如何使用 Streamlit 创建带有侧边栏的交互式应用。
        - 开发者：示例团队
        """
    )
    st.markdown("📄 [查看源代码](https://github.com/example/repo)")

# 页脚
st.markdown("---")
st.markdown("© 2024 示例团队")
