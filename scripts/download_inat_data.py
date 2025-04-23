import os
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm


def download_inat_user_images_with_metadata(
    username, country, date1, date2, save_dir="inat_images"
):
    """
    Download images and save metadata from iNaturalist observations by a specific user in a given country between two dates.

    Args:
        username (str): iNaturalist username.
        country (str): Country name (e.g., "Panama").
        date1 (str): Start date in 'YYYY-MM-DD' format.
        date2 (str): End date in 'YYYY-MM-DD' format.
        save_dir (str): Directory to save images and metadata.
    """
    os.makedirs(save_dir, exist_ok=True)
    metadata = []

    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "user_login": username,
        "place_guess": country,
        "d1": date1,
        "d2": date2,
        "per_page": 200,
        "page": 1,
        "order_by": "observed_on",
        "order": "asc",
    }

    print(
        f"Searching for observations by '{username}' in '{country}' from {date1} to {date2}..."
    )

    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            break

        data = response.json()
        results = data.get("results", [])
        if not results:
            print("No more results.")
            break

        for obs in tqdm(results, desc=f"Downloading page {params['page']}"):
            obs_id = obs.get("id")
            observed_on = obs.get("observed_on")
            latitude = obs.get("geojson", {}).get("coordinates", [None, None])[1]
            longitude = obs.get("geojson", {}).get("coordinates", [None, None])[0]
            taxon = obs.get("taxon", {})
            species_id = taxon.get("id")
            species_name = taxon.get("name")
            species_rank = taxon.get("rank")
            quality_grade = obs.get("quality_grade")

            if obs_id and obs.get("photos"):
                for idx, photo in enumerate(obs["photos"]):
                    photo_url = photo["url"].replace(
                        "square", "original"
                    )  # Get high-res
                    ext = os.path.splitext(photo_url)[1].split("?")[0]
                    filename = f"{obs_id}_{idx}{ext}"
                    filepath = os.path.join(save_dir, filename)

                    # Download image
                    try:
                        img_data = requests.get(photo_url).content
                        with open(filepath, "wb") as f:
                            f.write(img_data)
                    except Exception as e:
                        print(f"Failed to download {photo_url}: {e}")
                        continue

                    # Save metadata for each image
                    metadata.append(
                        {
                            "filename": filename,
                            "observation_id": obs_id,
                            "observed_on": observed_on,
                            "latitude": latitude,
                            "longitude": longitude,
                            "species_id": species_id,
                            "species_name": species_name,
                            "species_rank": species_rank,
                            "quality_grade": quality_grade,
                            "photo_url": photo_url,
                        }
                    )

        # Check if there are more pages
        if data["total_results"] <= params["page"] * params["per_page"]:
            break
        params["page"] += 1

    # Save metadata as CSV
    if metadata:
        df = pd.DataFrame(metadata)
        csv_path = os.path.join(save_dir, "metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metadata saved to {csv_path}")
    else:
        print("No metadata to save.")

    print(f"Download complete. Images and metadata saved to: {save_dir}")


# Example usage:
download_inat_user_images_with_metadata(
    username="mlarrivee",
    country="Panama",
    date1="2025-02-25",
    date2="2025-03-10",
    save_dir="/network/scratch/y/yuyan.chen/species_discovery/inat_totumas",
)
