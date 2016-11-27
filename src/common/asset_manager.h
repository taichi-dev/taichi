#pragma once

#include <cstring>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <iostream>

TC_NAMESPACE_BEGIN

class AssetManager {
public:
	// Note: this is not thread safe!
	template <typename T>
	std::shared_ptr<T> get_asset_(int id) {
		assert_info(id_to_asset.find(id) != id_to_asset.end(), "Asset not found");
		auto ptr = id_to_asset[id];
		assert(!ptr.expired());
		return std::static_pointer_cast<T>(ptr.lock());
	}

	template <typename T>
	int insert_asset_(const std::shared_ptr<T> &ptr) {
		assert_info(asset_to_id.find(ptr.get()) == asset_to_id.end(), "Asset already exists");
		int id = counter++;
		id_to_asset[id] = static_cast<std::weak_ptr<void>>(std::weak_ptr<T>(ptr));
		asset_to_id[ptr.get()] = id;
		return id;
	}

	template <typename T>
	static auto get_asset(int id) {
		return get_instance().get_asset_<T>(id);
	}

	template <typename T>
	static int insert_asset(const std::shared_ptr<T> &ptr) {
		return get_instance().insert_asset_<T>(ptr);
	}

	int counter = 0;
	std::map<void *, int> asset_to_id;
	std::map<int, std::weak_ptr<void>> id_to_asset;

	static AssetManager &get_instance() {
		static AssetManager manager;
		return manager;
	}
};

TC_NAMESPACE_END
