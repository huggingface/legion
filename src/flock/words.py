"""Word lists for readable agent names (`naming: words`, docs/spawn.md): `<adjective>-<noun>`,
lowercase letters only, so a name is a valid board handle. Colors and simple qualities;
animals, fruits, plants and everyday objects. Plain, neutral words only."""


def _list(text: str) -> tuple[str, ...]:
    return tuple(text.split())


ADJECTIVES = _list("""
amber azure beige black blue bronze brown coral cream crimson cyan golden gray green indigo
ivory jade khaki lemon lilac lime magenta maroon mint navy ochre olive orange peach pink plum
purple red rose ruby rust sage salmon sand scarlet silver sky slate snow tan teal violet white
yellow cobalt copper denim emerald fuchsia garnet honey lavender mauve mustard pearl sapphire
sepia smoky tawny topaz umber
able agile airy alert ample apt bold brave breezy bright brisk broad calm careful cheerful
chief civil clean clear clever close cool cosy crisp curious daring deft eager early easy
even fair fancy fast fine firm fleet fond free fresh friendly frank gentle glad good grand
great handy happy hardy hearty helpful honest humble jolly just keen kind lively loyal lucky
merry mild modern modest neat nimble noble patient plain plucky polite proud quick quiet rapid
ready regal robust round royal rugged safe sharp shiny simple sleek smart smooth snappy snug
soft solid sound spare speedy spry stable steady stout strong sturdy sunny super sure swift
tidy tiny tough trim true upbeat urban vivid warm wide wise witty young zesty zippy
""")

NOUNS = _list("""
ant badger bat bear beaver bee bison boar buffalo camel cat cheetah chicken cobra cod condor
cougar cow crab crane crow deer dingo dog dolphin donkey dove duck eagle eel elk emu falcon
ferret finch fox frog gazelle gecko goat goose gopher gorilla hare hawk hedgehog heron hippo
horse hyena ibis iguana jackal jaguar jay kangaroo kiwi koala lark lemur leopard lion lizard
llama lobster lynx magpie mink mole moose moth mouse mule newt ocelot octopus orca ostrich
otter owl ox panda panther parrot pelican penguin pigeon pony puffin puma quail rabbit raccoon
ram raven robin salmon seal shark sheep shrew skunk sloth snail sparrow squid stork swan tapir
tiger toad trout tuna turkey turtle viper walrus wasp weasel whale wolf wombat wren yak zebra
apple apricot avocado banana berry cherry coconut date fig grape guava kiwifruit lemon lime
lychee mango melon nectarine olive orange papaya peach pear pineapple plum pomelo quince
raisin tangerine
acorn almond aspen bamboo basil birch bluebell cactus cedar clover cypress daisy elm fern
hazel heather holly iris ivy jasmine juniper laurel lilac lily linden lotus maple mint moss
oak orchid palm pine poppy reed rose sage spruce thistle thyme tulip violet willow yew
anchor anvil apron arrow axe badge bag ball barrel basket beacon bell bench blanket boat bolt
book boot bottle bowl box brick broom brush bucket button cable camera candle canoe cap card
carpet cart chair chalk clock cloud coin comb compass cord cup curtain desk dial dish door
drum easel engine fan feather fence flag flute fork frame funnel gate glove hammer harp hat
helmet hinge hook jar kettle key kite ladder lamp lantern latch lens lever lock magnet map
marble mask mirror mitten mug nail needle net notebook oar paddle pail pan paper pen pencil
piano pillow pipe pitcher plate pocket pot quill radio rake ribbon ring rope ruler sail saw
scarf scissors screw shelf shield shovel sled spade spoon stamp stool stove string table
teapot tent thimble ticket tile torch towel tray trumpet umbrella vase violin wagon wallet
wheel whistle window wrench yarn zipper
""")
