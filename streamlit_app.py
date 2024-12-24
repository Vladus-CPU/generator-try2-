import streamlit as st
from transformers import pipeline

# Кэшируем загрузку модели, чтобы при каждом обновлении страницы
# модель не перезагружалась заново (ускоряет работу приложения)
@st.cache_resource
def load_text_generation_model():
    return pipeline("text-generation", model="distilgpt2")

def main():
    st.title("AI-генератор текста")
    st.markdown("""
    Введите начальный текст ниже и нажмите "Сгенерировать",
    чтобы увидеть продолжение, сформированное моделью.
    """)

    # Поле для ввода начального текста
    prompt = st.text_area("Начальный текст:", value="Пример: Я думаю, что")

    # Слайдер для выбора, сколько символов генерировать (longer → больше текста)
    max_length = st.slider("Длина генерируемого текста", 20, 200, 50)

    # Кнопка запуска генерации
    if st.button("Сгенерировать"):
        model = load_text_generation_model()
        output = model(prompt, max_length=max_length, num_return_sequences=1)
        
        # Вывод результата
        st.subheader("Результат:")
        st.write(output[0]['generated_text'])

if __name__ == "__main__":
    main()
