# Qwen Dataset — Text-to-Image Prompts

All 45 prompts used by `tools/generate_qwen_dataset.py` to generate the Atlas Camera evaluation dataset (`atlas_exports/qwen_dataset_01/`). Each image is generated once via Qwen-Image, then run through all 6 documented projection-derivation variants (`organic` / `indoor` / `outdoor` / `manual_azimuth` / `both` / `vertical_extrusion`).

Seeds are deterministic: `seed = BASE_SEED + index` where `BASE_SEED = 667099001500000` and `index` is the scene's 0-based position in the list below (so `exterior_01_apartment_sedan` = 667099001500000, `exterior_02_bus_stop` = 667099001500001, etc.).

## Exterior

### `exterior_01_apartment_sedan` (seed `667099001500000`)

photorealistic street-level photograph of a 5-story brick apartment building with five clearly countable rows of windows, a red sedan parked directly at the curb in front of it, one adult pedestrian walking past the car for scale, midday overcast light, straight-on eye-level camera height around 1.6m.

### `exterior_02_bus_stop` (seed `667099001500001`)

photorealistic photograph of a city bus stop, a standard city transit bus about 12 meters long parked at the curb, two adults waiting at the stop beside it, a 4-story office building behind them, late-afternoon sun casting long shadows.

### `exterior_03_highrise_construction` (seed `667099001500002`)

photorealistic photograph of a 5-story building under construction, steel scaffolding visible at every floor level, a yellow shipping-container-sized site office trailer beside it, one construction worker in a hi-vis vest and hard hat standing near the trailer for scale, overcast daylight.

### `exterior_04_suburban_house` (seed `667099001500003`)

photorealistic photograph of a two-story suburban house with a standard front door and a single-car attached garage, a compact SUV parked in the driveway, one adult retrieving mail from a mailbox at the curb for scale, bright sunny afternoon.

### `exterior_05_rainy_night_crosswalk` (seed `667099001500004`)

photorealistic rainy night street-crossing photograph, reflective wet asphalt, an adult pedestrian using a crosswalk beside a parked sedan, a traffic light and street lamps for scale, neon storefront signs in the background, cinematic dusk lighting.

### `exterior_06_double_decker` (seed `667099001500005`)

photorealistic photograph of a red double-decker tour bus, about 4.4 meters tall, stopped at a curb, one adult standing beside its front wheel for scale, a 5-story brick building facade in the background, morning golden-hour light.

### `exterior_07_gas_station` (seed `667099001500006`)

photorealistic photograph of a gas station forecourt, a standard sedan parked at a fuel pump, one adult filling the tank for scale, a 5-story hotel building visible behind the station, dusk lighting with the station canopy lights switched on.

### `exterior_08_school` (seed `667099001500007`)

photorealistic photograph of the front entrance of a 3-story brick schoolhouse, a standard flagpole about 6 meters tall beside the main door, a yellow school bus parked out front, two children walking toward the entrance for scale, midday sun.

### `exterior_09_parking_garage` (seed `667099001500008`)

photorealistic photograph of the entrance ramp to a multi-level parking garage, a height-clearance sign reading 2.1m mounted above the entrance, a mid-size sedan approaching the entrance, one adult attendant standing beside the booth for scale, overcast afternoon.

### `exterior_10_church` (seed `667099001500009`)

photorealistic photograph of a stone church with a bell tower roughly five stories tall, a standard arched wooden double door at the entrance, one adult in a coat walking up the front steps, a parked sedan at the curb, late-afternoon warm light.

### `exterior_11_alley_dumpster` (seed `667099001500010`)

photorealistic photograph of a narrow city alley between two 5-story brick buildings, a standard commercial dumpster about 1.8 meters long against one wall, one adult walking through the alley for scale, overcast daylight, fire escapes visible on the walls.

### `exterior_12_bridge` (seed `667099001500011`)

photorealistic photograph of a pedestrian crossing a steel truss bridge over a river, a standard guardrail about 1.1 meters tall along the walkway, a car crossing the bridge roadway lane beside the pedestrian path, midday clear sky.

### `exterior_13_storefront_row` (seed `667099001500012`)

photorealistic photograph of a row of ground-floor storefronts below 5 stories of apartments, awnings at a standard height above the sidewalk doors, one adult exiting a shop doorway, a delivery van double-parked at the curb, bright afternoon sun.

### `exterior_14_town_square` (seed `667099001500013`)

photorealistic photograph of a town square with a stone clock tower approximately five stories tall at its center, an ornate lamp post beside it, several adults walking across the square, a parked sedan at the square's edge, early-evening golden-hour light.

### `exterior_15_overpass` (seed `667099001500014`)

photorealistic photograph of a highway overpass with a standard passenger car driving beneath it, a pedestrian walking on the sidewalk beside the on-ramp, a highway sign mounted on the overpass, overcast midday light.

## Interior

### `interior_01_living_room` (seed `667099001500015`)

photorealistic photograph of the interior of a furnished living room, a standard 2032mm (6'8") interior door open in the background, an adult seated on a three-seat sofa in the foreground for scale, a coffee table in front of the sofa, warm afternoon window light.

### `interior_02_kitchen` (seed `667099001500016`)

photorealistic photograph of a modern kitchen with standard-height countertops about 90cm tall, a full-size refrigerator beside the counter, an adult standing at the stove for scale, overhead pendant lighting, straight-on eye-level camera angle.

### `interior_03_hotel_room` (seed `667099001500017`)

photorealistic photograph of a hotel room with a standard queen-size bed 152cm wide, a full-length door-mounted mirror on the closet door, one adult standing beside the bed for scale, warm lamp lighting, evening ambience.

### `interior_04_hallway` (seed `667099001500018`)

photorealistic photograph of a long apartment-building hallway lined with standard 2032mm interior doors at regular intervals, one adult walking down the hallway toward the camera for scale, fluorescent ceiling lights, straight-on perspective view.

### `interior_05_staircase` (seed `667099001500019`)

photorealistic photograph of a straight residential staircase with standard 7-inch risers and a handrail at standard height, one adult walking up the stairs for scale, daylight coming through a window at the landing.

### `interior_06_bathroom` (seed `667099001500020`)

photorealistic photograph of a standard bathroom with a full-size bathtub about 1.5 meters long and a pedestal sink, one adult standing beside the sink brushing their teeth for scale, tile flooring, bright overhead lighting.

### `interior_07_bedroom` (seed `667099001500021`)

photorealistic photograph of a bedroom with a standard full-size 2032mm door and a queen bed, one adult sitting on the edge of the bed for scale, a dresser with a mirror on the far wall, soft morning window light.

### `interior_08_restaurant` (seed `667099001500022`)

photorealistic photograph of a restaurant dining room with standard-height dining tables about 75cm tall and chairs, several adults seated at tables for scale, a server standing beside one table, warm ambient pendant lighting.

### `interior_09_gym` (seed `667099001500023`)

photorealistic photograph of an indoor gym with a standard 2032mm doorway and a rack of dumbbells, one adult standing beside a squat rack for scale, rubber flooring, bright fluorescent lighting.

### `interior_10_classroom` (seed `667099001500024`)

photorealistic photograph of a school classroom with rows of standard student desks about 75cm tall and chairs, a whiteboard at the front, a teacher standing beside the whiteboard for scale, daylight through tall windows.

### `interior_11_warehouse` (seed `667099001500025`)

photorealistic photograph of the interior of a warehouse with tall steel shelving racks about 4 meters high holding cardboard boxes, a forklift parked between the racks, one adult worker standing beside the forklift for scale, overhead industrial lighting.

### `interior_12_church_interior` (seed `667099001500026`)

photorealistic photograph of the interior of a church nave with tall wooden pews, standard bench height about 45cm, and a high vaulted ceiling, one adult seated in a pew for scale, stained-glass windows, soft daylight streaming in.

### `interior_13_retail_store` (seed `667099001500027`)

photorealistic photograph of a retail clothing store interior with standard clothing racks about 1.5 meters tall and a checkout counter, one adult customer browsing a rack for scale, a cashier at the counter, bright retail lighting.

### `interior_14_elevator_lobby` (seed `667099001500028`)

photorealistic photograph of a building elevator lobby with a standard 2032mm elevator door and a directory sign on the wall, one adult standing waiting for the elevator for scale, polished floor reflecting overhead lighting.

### `interior_15_office` (seed `667099001500029`)

photorealistic photograph of an open-plan office with standard cubicle partitions about 1.5 meters tall and desks, one adult seated at a desk typing for scale, a water cooler beside the wall, bright overhead fluorescent lighting.

## Nature

### `nature_01_forest_trail` (seed `667099001500030`)

photorealistic photograph of a forest trail, one adult hiker standing beside a large tree trunk for scale, dense pine forest in the background, dappled midday sunlight through the canopy.

### `nature_02_mountain_vista` (seed `667099001500031`)

photorealistic photograph of a hiker standing on a rocky overlook with a mountain range in the far distance, a standard backpack about 60cm tall on the ground beside them for scale, clear afternoon light.

### `nature_03_beach` (seed `667099001500032`)

photorealistic photograph of a sandy beach, one adult standing near the shoreline, a standard beach umbrella about 2 meters tall planted in the sand beside them, gentle waves in the background, bright midday sun.

### `nature_04_desert_dunes` (seed `667099001500033`)

photorealistic photograph of desert sand dunes, one adult walking along a dune ridge for scale, a parked SUV at the base of the dunes in the distance, clear blue sky, late-afternoon warm light.

### `nature_05_waterfall` (seed `667099001500034`)

photorealistic photograph of a waterfall cascading into a pool, one adult standing on rocks at the base of the falls for scale, mist rising from the pool, midday overcast light.

### `nature_06_lake_shore` (seed `667099001500035`)

photorealistic photograph of a calm lake shoreline with a wooden dock, one adult standing on the dock beside a canoe about 4.5 meters long pulled up on the shore, mountains reflected in the water, early-morning light.

### `nature_07_canyon` (seed `667099001500036`)

photorealistic photograph of a hiker standing at the rim of a canyon with layered rock walls descending below, a standard hiking pole about 1.3 meters long in hand for scale, midday clear light.

### `nature_08_meadow` (seed `667099001500037`)

photorealistic photograph of an open meadow with tall grass, one adult standing in the field with a golden retriever dog about 60cm tall at the shoulder beside them for scale, distant tree line, soft late-afternoon light.

### `nature_09_riverbank` (seed `667099001500038`)

photorealistic photograph of a riverbank with smooth stones, one adult crouched beside the water fishing, a standard kayak about 3 meters long resting on the bank beside them, midday light with gentle reflections.

### `nature_10_snowy_slope` (seed `667099001500039`)

photorealistic photograph of a snowy mountain slope, one skier standing at the top for scale, evergreen trees dotting the slope below, bright clear winter daylight.

### `nature_11_rocky_coastline` (seed `667099001500040`)

photorealistic photograph of a rocky ocean coastline with crashing waves, one adult standing on a flat rock outcrop for scale, a lighthouse visible in the distance, overcast dramatic sky.

### `nature_12_redwood_forest` (seed `667099001500041`)

photorealistic photograph of a towering redwood forest, one adult standing at the base of a massive redwood trunk for scale, ferns covering the forest floor, soft filtered light through the canopy.

### `nature_13_farmland_silo` (seed `667099001500042`)

photorealistic photograph of open farmland with a grain silo about five stories (15 meters) tall beside a red barn, a pickup truck parked near the barn, one farmer standing beside the truck for scale, golden late-afternoon light.

### `nature_14_botanical_garden` (seed `667099001500043`)

photorealistic photograph of a botanical garden path lined with flowering shrubs, one adult walking along the path beside a stone garden bench about 1.5 meters long, midday bright sunlight.

### `nature_15_cliffside_path` (seed `667099001500044`)

photorealistic photograph of a narrow cliffside hiking path along a coastal bluff, one adult hiker walking the path with the ocean far below for scale, a wooden trail marker post about 1.2 meters tall beside the path, clear afternoon light.
