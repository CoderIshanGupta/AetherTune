import auth

def recommend(activity: int):

    if auth.spotify_client is None:
        return []

    activity_queries = {
        0: "lofi study instrumental focus music",
        1: "road trip upbeat pop driving songs",
        2: "calm meditation ambient relaxing music",
        3: "high energy workout gym edm motivation"
    }

    query = activity_queries.get(activity)

    if query is None:
        return []

    try:
        results = auth.spotify_client.search(
            q=query,
            type="track",
            limit=5
        )

        tracks = results["tracks"]["items"]

        recommendations = []

        for track in tracks:
            recommendations.append({
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "id": track["id"],                 # important
                "url": track["external_urls"]["spotify"]
            })

        return recommendations

    except Exception as e:
        print("Recommendation Error:", e)
        return []