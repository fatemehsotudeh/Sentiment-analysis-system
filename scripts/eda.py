from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_word_cloud(df, sentiment):
    word_cloud_text = ''.join(df[df["sentiment"]==sentiment].lower)

    wordcloud = WordCloud(
        max_font_size=100,
        max_words=100,
        background_color="black",
        scale=10,
        width=800,
        height=800
    ).generate(word_cloud_text)

    plt.figure(figsize=(10,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()