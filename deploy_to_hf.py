import os
import sys
# Fix Windows console encoding
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
from huggingface_hub import HfApi, login

def main():
    print("====================================================")
    print("      CineCut AI - HuggingFace Spaces Auto-Deployer ")
    print("====================================================")
    
    token = input("Paste token: ").strip()
    if not token:
        print("Token cannot be empty!")
        return

    try:
        login(token=token)
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']
        print(f"Logged in successfully as: {username}")
        
        repo_id = f"{username}/cinecut-ai-backend"
        print(f"Creating Space: {repo_id}...")
        
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True
        )
        print("Space created successfully!")
        
        files_to_upload = ["server.py", "Dockerfile", "requirements.txt"]
        for f in files_to_upload:
            if os.path.exists(f):
                print(f"Uploading {f}...")
                api.upload_file(
                    path_or_fileobj=f,
                    path_in_repo=f,
                    repo_id=repo_id,
                    repo_type="space"
                )
        
        app_url = f"https://{username}-cinecut-ai-backend.hf.space"
        print("Upload complete!")
        
        # Update app.js automatically with the new Hugging Face Space URL!
        app_js_path = "app.js"
        if os.path.exists(app_js_path):
            with open(app_js_path, "r", encoding="utf-8") as file:
                content = file.read()
            # Replace Render URL with Hugging Face Space URL
            content = content.replace("https://cinecut-ai-backend.onrender.com", app_url)
            with open(app_js_path, "w", encoding="utf-8") as file:
                file.write(content)
            print("Updated app.js with HuggingFace Space URL!")
            
            # Push changes to GitHub/Vercel
            os.system("git add app.js && git commit -m \"Connect Vercel to HuggingFace Space\" && git push origin master")
            print("Pushed updated app.js to GitHub/Vercel!")
            
        print("ALL SETUP COMPLETE!")
        print(f"Your AI Cloud Server: {app_url}")
        print("====================================================")
        
    except Exception as e:
        print(f"Error during deployment: {e}")

if __name__ == "__main__":
    main()
