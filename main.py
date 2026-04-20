from utils.file_manager import FileManager

def main() -> None:
    file_manager = FileManager()
    data = file_manager.load_pickle("faiss_index/index.pkl")
    print()


if __name__ == "__main__":
    main()

