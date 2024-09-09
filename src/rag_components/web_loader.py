from langchain_community.document_loaders import WebBaseLoader
def get_webdata():
    webpage_loader = WebBaseLoader(["https://www.brainwired.in/about-us","https://www.brainwired.in/","https://www.brainwired.in/blog","https://www.brainwired.in/career","https://www.brainwired.in/our-team",
                        ])
    web_document = webpage_loader.load()
    return web_document