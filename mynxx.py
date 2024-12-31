if __name__ == "__main__":
    # 出力文字列ヘッダを生成
    print("\n\n")
    print("#"*100)
    print("Listening for wakewords...")
    print("#"*100)
    print("\n"*(n_models*3))

    while True:
        # 音声を取得
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # 音声をopenWakeWordモデルに渡す
        prediction = owwModel.predict(audio)

        # タイトル列
        n_spaces = 16
        output_string_header = """
            Model Name         | Score | Wakeword Status
            --------------------------------------
            """

        for mdl in owwModel.prediction_buffer.keys():
            # フォーマットされた表にスコアを追加
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")

            output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
            """

        # 結果を表で出力
        print("\033[F"*(4*n_models+1))
        print(output_string_header, "                             ", end='\r')
