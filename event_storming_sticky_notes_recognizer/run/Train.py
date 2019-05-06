from event_storming_sticky_notes_recognizer.run.models.ModelLoader import ModelLoader

if __name__ == "__main__":
    loader = ModelLoader()
    model = loader.load(name='simple_net')
    print(model)
