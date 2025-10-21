import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import model_utils as mu

st.set_page_config(layout="wide")
st.title("🚨 Cyber Threat Detection Based on Artificial Neural Networks Using Event Profiles")

# Initialize session state variables
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'X_train' not in st.session_state:
    st.session_state.X_train = st.session_state.X_test = st.session_state.y_train = st.session_state.y_test = None
if 'num_classes' not in st.session_state:
    st.session_state.num_classes = None
if 'df' not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    if st.button("Run Preprocessing TF-IDF Algorithm"):
        with st.spinner("Running TF-IDF preprocessing..."):
            tfidf = mu.tfidf_preprocess(df)
            st.subheader("✅ TF-IDF Matrix")
            st.write(pd.DataFrame(tfidf.toarray()))

    if st.button("Generate Event Vector"):
        with st.spinner("Generating Event Vector..."):
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test, st.session_state.num_classes = mu.event_vector_generation(df)
        st.success("Event Vector Generated Successfully!")

    if st.button("Neural Network Profiling"):
        with st.spinner("Running ANN Model..."):
            acc, prec, rec, f1 = mu.run_ann(
                st.session_state.X_train, st.session_state.y_train,
                st.session_state.X_test, st.session_state.y_test,
                st.session_state.num_classes
            )
        st.session_state.model_results['ANN'] = (acc, prec, rec, f1)
        st.success("ANN Completed!")
        st.markdown("**ANN Results:**")
        st.write(f"Accuracy: {acc:.2f}%")
        st.write(f"Precision: {prec:.2f}%")
        st.write(f"Recall: {rec:.2f}%")
        st.write(f"F1-Score: {f1:.2f}%")

    model_buttons = {
        "Run KNN Algorithm": mu.run_knn,
        "Run SVM Algorithm": mu.run_svm,
        "Run Decision Tree Algorithm": mu.run_dt,
        "Run Random Forest Algorithm": mu.run_rf,
        "Run Naive Bayes Algorithm": mu.run_nb
    }

    for label, func in model_buttons.items():
        if st.button(label):
            with st.spinner(f"Running {label}..."):
                acc, prec, rec, f1 = func(
                    st.session_state.X_train, st.session_state.y_train,
                    st.session_state.X_test, st.session_state.y_test
                )
                name = label.replace("Run ", "").replace(" Algorithm", "")
                st.session_state.model_results[name] = (acc, prec, rec, f1)
            st.success(f"{name} Completed!")
            st.markdown(f"**{name} Results:**")
            st.write(f"Accuracy: {acc:.2f}%")
            st.write(f"Precision: {prec:.2f}%")
            st.write(f"Recall: {rec:.2f}%")
            st.write(f"F1-Score: {f1:.2f}%")

    if st.session_state.model_results:
        def plot_comparison(metric_index, title):
            fig, ax = plt.subplots()
            models = list(st.session_state.model_results.keys())
            values = [st.session_state.model_results[m][metric_index] for m in models]
            ax.bar(models, values, color='skyblue')
            ax.set_ylabel(title)
            ax.set_title(f"{title} Comparison")
            ax.set_ylim([0, 100])
            st.pyplot(fig)

        if st.button("Accuracy Comparison Graph"):
            plot_comparison(0, "Accuracy")

        if st.button("Precision Comparison Graph"):
            plot_comparison(1, "Precision")

        if st.button("Recall Comparison Graph"):
            plot_comparison(2, "Recall")

        if st.button("FMeasure Comparison Graph"):
            plot_comparison(3, "F1-Score")
