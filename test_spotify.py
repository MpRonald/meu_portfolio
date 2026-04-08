from services import get_spotify_token, search_music

print("TOKEN:")
print(get_spotify_token())

print("\nSEARCH:")
print(search_music("jazz"))