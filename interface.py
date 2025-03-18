import os
from query_system import PDFMemorySystem

def interactive_cli(pdf_memory, log_file="conversations/chat_log.txt"):
    """Interactive command-line interface for the PDF Memory System."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    print("\n" + "="*50)
    print("PDF Memory System CLI")
    print("="*50 + "\n")
    print("Type your questions below and press Enter.")
    print("Type 'quit', 'exit', or 'q' to exit.")
    
    while True:
        try:
            question = input("\nYour question: ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the PDF Memory System. Goodbye!")
                break
            
            print("\nSearching through documents...\n")
            answer = pdf_memory.query(question)
            
            print(f"Answer: {answer}\n")
            pdf_memory.save_conversation(question, answer, log_file)
            
        except KeyboardInterrupt:
            print("\nOperation interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

def create_web_ui(pdf_memory, server_port=7860):
    """Create a web interface using Gradio."""
    import gradio as gr
    
    # Define interface function
    def process_query(question, history):
        if not question:
            return "", history
        
        response = pdf_memory.query(question)
        history.append((question, response))
        pdf_memory.save_conversation(question, response, "conversations/web_log.txt")
        return "", history
    
    # Create interface
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("# PDF Memory System")
        gr.Markdown("Ask questions about your PDF documents and get answers with source attribution.")
        
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(placeholder="Type your question here...", show_label=False)
        clear = gr.Button("Clear Chat")
        
        msg.submit(process_query, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot)
    
    # Launch interface
    demo.launch(server_port=server_port, share=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch PDF Memory System interface")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--vector_db_path", type=str, required=True, help="Path to vector database")
    parser.add_argument("--interface", type=str, choices=["cli", "web"], default="cli", 
                        help="Interface type (cli or web)")
    parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    
    args = parser.parse_args()
    
    # Initialize the PDF Memory System
    system = PDFMemorySystem(args.model_path, args.vector_db_path)
    
    # Launch the selected interface
    if args.interface == "cli":
        interactive_cli(system)
    else:
        create_web_ui(system, args.port)
