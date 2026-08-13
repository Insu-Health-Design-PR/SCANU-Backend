"""Model profile CRUD and apply."""

# TODO: load/save config/profiles/model_profiles.json


class ProfileService:
    def list_profiles(self) -> list[dict]:
        return []

    def apply(self, profile_id: str) -> dict:
        raise NotImplementedError
